import os
import sys
import math
import copy
import collections
import numpy as np

from io import open
from typing import Callable, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from transformers import BertPreTrainedModel, PretrainedConfig
sys.path.append(os.path.dirname(__file__))
from lavis.models import load_model

from .ops import create_transformer_encoder
from .ops import extend_neg_masks, gen_seq_masks, pad_tensors_wgrad
from .vilmodel import BertAttention, BertIntermediate, BertOutput, BertXAttention

BertLayerNorm = torch.nn.LayerNorm

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)

class ClsPrediction(nn.Module):
    def __init__(self, hidden_size, input_size=None):
        super().__init__()
        if input_size is None:
            input_size = hidden_size
        self.net = nn.Sequential(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(),
                                 BertLayerNorm(hidden_size, eps=1e-12),
                                 nn.Linear(hidden_size, 1))

    def forward(self, x):
        return self.net(x)

class GraphLayer(nn.Module):
    '''
    Graph Layer without crossmodal attention with language
    '''
    def __init__(self, config):
        super().__init__()

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

    def forward(
        self, visn_feats, visn_attention_mask,
        graph_sprels=None
    ):

        if graph_sprels is not None:
            visn_attention_mask = visn_attention_mask + graph_sprels
        visn_att_output = self.visn_self_att(visn_feats, visn_attention_mask)[0]

        visn_inter_output = self.visn_inter(visn_att_output)
        visn_output = self.visn_output(visn_inter_output, visn_att_output)

        return visn_output

class GraphLXRTXLayer(nn.Module):
    '''
    Graph Layer with crossmodal attention with language
    '''
    def __init__(self, config):
        super().__init__()

        # Lang self-att and FFN layer
        if config.use_lang2visn_attn:
            self.lang_self_att = BertAttention(config)
            self.lang_inter = BertIntermediate(config)
            self.lang_output = BertOutput(config)

        # Visn self-att and FFN layer
        self.visn_self_att = BertAttention(config)
        self.visn_inter = BertIntermediate(config)
        self.visn_output = BertOutput(config)

        # The cross attention layer
        self.visual_attention = BertXAttention(config, ctx_dim=config.llm_ctx_size)

    def forward(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
        graph_sprels=None
    ):      
        visn_att_output = self.visual_attention(
            visn_feats, lang_feats, ctx_att_mask=lang_attention_mask
        )[0]

        if graph_sprels is not None:
            visn_attention_mask = visn_attention_mask + graph_sprels
        visn_att_output = self.visn_self_att(visn_att_output, visn_attention_mask)[0]

        visn_inter_output = self.visn_inter(visn_att_output)
        visn_output = self.visn_output(visn_inter_output, visn_att_output)

        return visn_output

    def forward_lang2visn(
        self, lang_feats, lang_attention_mask, visn_feats, visn_attention_mask,
    ):
        lang_att_output = self.visual_attention(
            lang_feats, visn_feats, ctx_att_mask=visn_attention_mask
        )[0]
        lang_att_output = self.lang_self_att(
            lang_att_output, lang_attention_mask
        )[0]
        lang_inter_output = self.lang_inter(lang_att_output)
        lang_output = self.lang_output(lang_inter_output, lang_att_output)
        return lang_output


class GASAEncoder(nn.Module):
    '''
    Graph aware self-attention encoder
    '''
    def __init__(self, config):
        super().__init__()
        self.num_x_layers = config.num_x_layers
        self.x_layers = nn.ModuleList(
            [GraphLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, img_embeds, img_masks, graph_sprels=None):
        extended_img_masks = extend_neg_masks(img_masks) # (N, 1(H), 1(L_q), L_v)
        for layer_module in self.x_layers:
            img_embeds = layer_module(
                img_embeds, extended_img_masks,
                graph_sprels=graph_sprels
            )
        return img_embeds

class GACAEncoder(nn.Module):
    '''
    Graph aware cross-attention encoder
    '''
    def __init__(self, config):
        super().__init__()
        self.num_x_layers = config.num_x_layers
        self.x_layers = nn.ModuleList(
            [GraphLXRTXLayer(config) for _ in range(self.num_x_layers)]
        )

    def forward(self, txt_embeds, txt_masks, img_embeds, img_masks, graph_sprels=None):
        extended_txt_masks = extend_neg_masks(txt_masks)
        extended_img_masks = extend_neg_masks(img_masks) # (N, 1(H), 1(L_q), L_v)
        for layer_module in self.x_layers:
            img_embeds = layer_module(
                txt_embeds, extended_txt_masks, 
                img_embeds, extended_img_masks,
                graph_sprels=graph_sprels
            )
        return img_embeds

class ImageEmbeddings(BertPreTrainedModel):
    def __init__(self, args):
        config = PretrainedConfig.from_pretrained('bert-base-uncased')
        # Now, iterate over the arguments and update the pre_config
        for arg in vars(args):
            value = getattr(args, arg)
            setattr(config, arg, value)
        super().__init__(config)

        self.single_action_head = config.single_action_head
        self.vp_proj = nn.Sequential(
            nn.Linear(config.llm_ctx_size, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        if config.llm_token_merge == 'linear':
            self.llm_token_merge = 'linear'
            self.token_merge = nn.Conv1d(in_channels=32, out_channels=1, kernel_size=1, bias=True)
        elif config.llm_token_merge == 'mean':
            self.llm_token_merge = 'mean'
            self.token_merge = nn.AvgPool1d(32)

        # merging CLIP CLS token and LLM image embedding
        self.use_clip_img_emb = config.use_clip_img_emb
        if self.use_clip_img_emb:
            self.img_linear = nn.Linear(config.image_feat_size + config.hidden_size, config.hidden_size)
        else:
            self.img_linear = nn.Linear(config.hidden_size, config.hidden_size)
        self.img_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.loc_linear = nn.Linear(2*(config.angle_feat_size + 3), config.hidden_size)
        self.loc_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)

        if config.obj_feat_size > 0 and config.obj_feat_size != config.image_feat_size:
            self.obj_linear = nn.Linear(config.obj_feat_size, config.hidden_size)
            self.obj_layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        else:
            self.obj_linear = self.obj_layer_norm = None

        # 0: navigable viewpoint, 1: object
        self.nav_type_embedding = nn.Embedding(2, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(2, config.hidden_size)

        # tf naming convention for layer norm
        self.layer_norm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if config.num_pano_layers > 0:
            self.pano_encoder = create_transformer_encoder(
                config, config.num_pano_layers, norm=True
            )
        else:
            self.pano_encoder = None
        
        self.init_weights()

    def forward(self, mode, batch, **kwargs):
        # forward sap in pretrain mode
        if mode == 'sap':
            return self.forward_sap(
                batch['traj_view_img_fts'], batch['traj_obj_img_fts'], batch['traj_loc_fts'],
                batch['traj_nav_types'], batch['traj_step_lens'], batch['traj_vp_view_lens'],
                batch['traj_vp_obj_lens']
            )
        # forward in inference or finetune mode
        elif mode == 'per_step':
            return self.forward_per_step(
                batch['view_cls_fts'], batch['view_llm_fts'], batch['obj_img_fts'], batch['loc_fts'],
                batch['nav_types'], batch['view_lens'], batch['obj_lens']
            )

    def forward_sap(
        self, traj_view_img_fts, traj_obj_img_fts, traj_loc_fts, traj_nav_types, 
        traj_step_lens, traj_vp_view_lens, traj_vp_obj_lens
    ):
        device = traj_view_img_fts.device
        has_obj = traj_obj_img_fts is not None

        traj_view_img_embeds = self.img_layer_norm(self.img_linear(traj_view_img_fts))
        if has_obj:
            if self.obj_linear is None:
                traj_obj_img_embeds = self.img_layer_norm(self.img_linear(traj_obj_img_fts))
            else:
                traj_obj_img_embeds = self.obj_layer_norm(self.obj_linear(traj_obj_img_embeds))
            traj_img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                traj_view_img_embeds, traj_obj_img_embeds, traj_vp_view_lens, traj_vp_obj_lens
            ):
                if obj_len > 0:
                    traj_img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    traj_img_embeds.append(view_embed[:view_len])
            traj_img_embeds = pad_tensors_wgrad(traj_img_embeds)
            traj_vp_lens = traj_vp_view_lens + traj_vp_obj_lens
        else:
            traj_img_embeds = traj_view_img_embeds
            traj_vp_lens = traj_vp_view_lens

        traj_embeds = traj_img_embeds + \
                      self.loc_layer_norm(self.loc_linear(traj_loc_fts)) + \
                      self.nav_type_embedding(traj_nav_types) + \
                      self.token_type_embeddings(torch.ones(1, 1).long().to(device))
        traj_embeds = self.layer_norm(traj_embeds)
        traj_embeds = self.dropout(traj_embeds)

        traj_masks = gen_seq_masks(traj_vp_lens)
        if self.pano_encoder is not None:
            traj_embeds = self.pano_encoder(
                traj_embeds, src_key_padding_mask=traj_masks.logical_not()
            )

        split_traj_embeds = torch.split(traj_embeds, traj_step_lens, 0)
        split_traj_vp_lens = torch.split(traj_vp_lens, traj_step_lens, 0)
        return split_traj_embeds, split_traj_vp_lens
    
    def forward_per_step(
        self, view_cls_fts, view_llm_fts, obj_img_fts, loc_fts, nav_types, view_lens, obj_lens
    ):
        device = view_llm_fts.device
        has_obj = obj_img_fts is not None

        # project view llm image tokens to hidden size
        view_llm_fts = self.vp_proj(view_llm_fts)                           # (N * 32, C)
        n_batch_views = view_llm_fts.size(0) // 32

        # merge 32 view image tokens for each view
        if self.llm_token_merge == 'linear':
            view_llm_fts = view_llm_fts.reshape(n_batch_views, 32, -1)      # (N, 32, C)
            view_llm_fts = self.token_merge(view_llm_fts).squeeze(1)        # (N, C)
        elif self.llm_token_merge == 'mean':
            view_llm_fts = view_llm_fts.reshape(n_batch_views, -1, 32)      # (N, C, 32)
            view_llm_fts = self.token_merge(view_llm_fts).squeeze(-1)       # (N, C)

        view_llm_fts = torch.split(view_llm_fts, view_lens.tolist(), 0)
        view_llm_fts = pad_tensors_wgrad(view_llm_fts)                      # (B, N_max, C)

        if self.use_clip_img_emb:
            view_img_fts = torch.cat([view_cls_fts, view_llm_fts], 2)
        else:
            view_img_fts = view_llm_fts

        view_img_embeds = self.img_layer_norm(
            self.img_linear(view_img_fts)
        )
        if has_obj:
            if self.obj_linear is None:
                obj_img_embeds = self.img_layer_norm(
                    self.img_linear(obj_img_fts)
                )
            else:
                obj_img_embeds = self.obj_layer_norm(
                    self.obj_linear(obj_img_fts)
                )
            img_embeds = []
            for view_embed, obj_embed, view_len, obj_len in zip(
                view_img_embeds, obj_img_embeds, view_lens, obj_lens
            ):
                if obj_len > 0:
                    img_embeds.append(torch.cat([view_embed[:view_len], obj_embed[:obj_len]], 0))
                else:
                    img_embeds.append(view_embed[:view_len])
            img_embeds = pad_tensors_wgrad(img_embeds)
            pano_lens = view_lens + obj_lens
        else:
            img_embeds = view_img_embeds
            pano_lens = view_lens

        if not self.single_action_head:
            # view image embeddings only if single action head is not used
            pano_embeds = img_embeds + \
                          self.loc_layer_norm(self.loc_linear(loc_fts.reshape(img_embeds.shape[0], -1, 14))) + \
                          self.token_type_embeddings(torch.ones(1, 1).long().to(device))
                        #   self.nav_type_embedding(nav_types) + \
            pano_embeds = self.layer_norm(pano_embeds)
        else:
            pano_embeds = img_embeds
        pano_embeds = self.dropout(pano_embeds)

        pano_masks = gen_seq_masks(pano_lens)
        if self.pano_encoder is not None:
            pano_embeds = self.pano_encoder(
                pano_embeds, src_key_padding_mask=pano_masks.logical_not()
            )
        return pano_embeds, pano_masks

class NavGPTThought(BertPreTrainedModel):
    def __init__(self, args):
        config = PretrainedConfig.from_pretrained('bert-base-uncased')
        # Now, iterate over the arguments and update the pre_config
        for arg in vars(args):
            value = getattr(args, arg)
            setattr(config, arg, value)
        super().__init__(config)

        self.vp_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size*2 + 6, 768),
            BertLayerNorm(768, eps=1e-12)
        )

        self.init_weights()

        self.Blip2InstructNav = load_model(
            name=config.arch,                       # 'blip2_t5_instruct_nav'
            model_type=config.model_type,           # 'flant5xl'
            checkpoint=config.qformer_ckpt_path,    # load from stage1 pretrained checkpoint
        )

        if config.load_patch_feature:
            del self.Blip2InstructNav.visual_encoder
        
        if not config.output_thought:
            if config.model_type.startswith('flant5'):
                del self.Blip2InstructNav.t5_model.decoder
                del self.Blip2InstructNav.t5_model.lm_head
                print("[INFO] Removed T5 decoder and LM head to save memory during training.")

        # Blip2InstructNav config
        self.model_type = config.model_type
        self.load_patch_feature = config.load_patch_feature
        self.output_thought = config.output_thought
        self.qformer_pos_emb = config.qformer_pos_emb
        # Generation config
        self.use_nucleus_sampling = config.use_nucleus_sampling
        self.num_beams = config.num_beams
        self.max_length = config.max_length
        self.min_length = config.min_length
        self.repetition_penalty = config.repetition_penalty
        self.length_penalty = config.length_penalty
        self.num_captions = config.num_captions
        self.top_p = config.top_p
        self.temperature = config.temperature

    def forward(
        self, images, loc_feats, qformer_text_inputs, text_inputs, text_outputs,
    ):
        # Q-former forward
        with self.Blip2InstructNav.maybe_autocast():
            if self.load_patch_feature:
                image_embeds = self.Blip2InstructNav.ln_vision(images)
            else:
                image_embeds = self.Blip2InstructNav.ln_vision(self.Blip2InstructNav.visual_encoder(images))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(images.device)

        query_tokens = self.Blip2InstructNav.query_tokens.expand(image_embeds.shape[0], -1, -1)

        # positional embedding
        if self.qformer_pos_emb:
            vp_pos_embeds = self.vp_pos_embeddings(loc_feats).unsqueeze(1)
            query_tokens = query_tokens + vp_pos_embeds

        if self.Blip2InstructNav.qformer_text_input:
            text_Qformer = self.Blip2InstructNav.tokenizer(
                qformer_text_inputs,
                padding='longest',
                truncation=True,
                max_length=self.Blip2InstructNav.max_txt_len,
                return_tensors="pt",
            ).to(images.device)
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(images.device)
            Qformer_atts = torch.cat([query_atts,text_Qformer.attention_mask[:, :self.Blip2InstructNav.Qformer.config.max_position_embeddings]],dim=1)

            query_output = self.Blip2InstructNav.Qformer.bert(
                text_Qformer.input_ids[:, :self.Blip2InstructNav.Qformer.config.max_position_embeddings],
                attention_mask=Qformer_atts,
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        else:
            query_output = self.Blip2InstructNav.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

        # SECTION 1: Q-former query projection
        if self.model_type.startswith('flant5'):
            # Project to T5 embedding space
            inputs_llm = self.Blip2InstructNav.t5_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])
        else:
            # Project to Vicuna embedding space
            inputs_llm = self.Blip2InstructNav.llm_proj(query_output.last_hidden_state[:,:query_tokens.size(1),:])

        # SECTION 2: input text tokenization
        with self.Blip2InstructNav.maybe_autocast(dtype=torch.bfloat16):
            if self.model_type.startswith('flant5'):
                # T5 tokenization
                input_tokens = self.Blip2InstructNav.t5_tokenizer(
                    text_inputs,
                    padding="longest",
                    truncation=True,
                    max_length=self.Blip2InstructNav.max_txt_len,
                    return_tensors="pt",
                ).to(images.device)

                # Calculate output target if Navigation thoughts provided
                if text_outputs is not None:
                    output_tokens = self.Blip2InstructNav.t5_output_tokenizer(
                        text_outputs,
                        padding="longest",
                        truncation=True,
                        max_length=self.Blip2InstructNav.max_output_txt_len,
                        return_tensors="pt",
                    ).to(images.device)

                    targets = output_tokens.input_ids.masked_fill(
                        output_tokens.input_ids == self.Blip2InstructNav.t5_tokenizer.pad_token_id, -100
                    )
                else:
                    targets = None
                
                encoder_atts = input_tokens.attention_mask
                
                inputs_embeds = self.Blip2InstructNav.t5_model.encoder.embed_tokens(input_tokens.input_ids)
            else:
                # Vicuna tokenization
                self.Blip2InstructNav.llm_tokenizer.padding_side = "right"
                self.Blip2InstructNav.llm_tokenizer.truncation_side = 'left'
                input_tokens = self.Blip2InstructNav.llm_tokenizer(
                    text_inputs,
                    return_tensors="pt",
                    padding="longest",
                    truncation=True,
                    max_length=self.Blip2InstructNav.max_txt_len,
                ).to(images.device)

                if text_outputs is not None:
                    # Tokenize output text
                    self.Blip2InstructNav.llm_tokenizer.truncation_side = 'right'
                    text_output_tokens = self.Blip2InstructNav.llm_tokenizer(
                        [t + self.Blip2InstructNav.llm_tokenizer.eos_token for t in text_outputs],
                        return_tensors="pt",
                        padding="longest",
                        truncation=True,
                        max_length=self.Blip2InstructNav.max_output_txt_len,
                    ).to(images.device)

                    llm_tokens, input_part_targets_len = self.Blip2InstructNav.concat_text_input_output(
                        input_tokens.input_ids,
                        input_tokens.attention_mask,
                        text_output_tokens.input_ids,
                        text_output_tokens.attention_mask,
                    )
                    # do not apply loss to the padding
                    targets = llm_tokens['input_ids'].masked_fill(
                        llm_tokens['input_ids'] == self.Blip2InstructNav.llm_tokenizer.pad_token_id, -100
                    )

                    # do not apply loss to the text input (i.e., instruction)
                    for i, l in enumerate(input_part_targets_len):
                        targets[i][:l] = -100

                    inputs_embeds = self.Blip2InstructNav.llm_model.get_input_embeddings()(llm_tokens['input_ids'])
                    encoder_atts = llm_tokens['attention_mask']
                else:
                    # No output text provided
                    # do not apply loss to the padding
                    targets = input_tokens.input_ids.masked_fill(
                        input_tokens.input_ids == self.Blip2InstructNav.llm_tokenizer.pad_token_id, -100
                    )
                    inputs_embeds = self.Blip2InstructNav.llm_model.get_input_embeddings()(input_tokens.input_ids)
                    encoder_atts = input_tokens.attention_mask
                
            # SECTION 3: Replace image tokens with image features
            if self.model_type.startswith('flant5'):
                # Get instruction tokens
                self.instruct_start_id = self.Blip2InstructNav.t5_tokenizer.convert_tokens_to_ids(['[INST]'])[0]
                self.instruct_end_id = self.Blip2InstructNav.t5_tokenizer.convert_tokens_to_ids(['[/INST]'])[0]

                # Get image tokens
                self.image_token_id = self.Blip2InstructNav.t5_tokenizer.convert_tokens_to_ids(['<image>'])[0]
            else:
                # Get instruction tokens
                self.instruct_start_id = self.Blip2InstructNav.llm_tokenizer.convert_tokens_to_ids(['[INST]'])[0]
                self.instruct_end_id = self.Blip2InstructNav.llm_tokenizer.convert_tokens_to_ids(['[/INST]'])[0]

                # Get image tokens
                self.image_token_id = self.Blip2InstructNav.llm_tokenizer.convert_tokens_to_ids(['<image>'])[0]
            
            all_image_indices = (input_tokens.input_ids == self.image_token_id).to(inputs_llm.device)

            assert (input_tokens.input_ids[all_image_indices].shape[0] == inputs_llm.shape[0] * inputs_llm.shape[1]), \
            f"image tokens in input ids {input_tokens.input_ids[input_tokens.input_ids == self.image_token_id].shape[0]} != the number of image tokens {inputs_llm.shape[0]}*{inputs_llm.shape[1]}"
            assert (inputs_llm.shape[-1] == inputs_embeds.shape[-1]), f"{inputs_llm.shape[-1]} != {inputs_embeds.shape[-1]}"

            inputs_llm = inputs_llm.reshape(-1, inputs_llm.shape[-1]).to(inputs_embeds.dtype)
            inputs_embeds[all_image_indices] = inputs_llm

            # SECTION 4: Forward LLM model
            output_text = None
            loss = None
            if self.model_type.startswith('flant5'):
                if targets is not None:
                    # Calculate caption loss if Navigation thoughts provided
                    outputs = self.Blip2InstructNav.t5_model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=encoder_atts,
                        decoder_attention_mask=output_tokens.attention_mask,
                        return_dict=True,
                        labels=targets,
                    )
                    loss = outputs.loss
                    encoder_last_hidden_state = outputs.encoder_last_hidden_state
                else:
                    # Forward encoder only if Navigation thoughts not provided
                    encoder_outputs = self.Blip2InstructNav.t5_model.encoder(
                        input_ids=None,
                        attention_mask=encoder_atts,
                        inputs_embeds=inputs_embeds,
                        head_mask=None,
                        output_attentions=None,
                        output_hidden_states=None,
                        return_dict=True,
                    )
                    encoder_last_hidden_state = encoder_outputs.last_hidden_state
            else:
                # Forward entire model for decoder only Llama model
                outputs = self.Blip2InstructNav.llm_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=encoder_atts,
                    return_dict=True,
                    output_hidden_states=True,
                    labels=targets,
                )
                loss = outputs.loss
                encoder_last_hidden_state = outputs.hidden_states[-1]
            
            view_embeds = encoder_last_hidden_state[all_image_indices,:] # (N * 32, C), all image tokens in batch are stacked together
            # Extract instruction embedding
            instruct_start_indices = torch.where(input_tokens.input_ids == self.instruct_start_id)
            instruct_end_indices = torch.where(input_tokens.input_ids == self.instruct_end_id)
            assert (len(instruct_start_indices[0]) == len(instruct_end_indices[0])), \
            f"number of instruction start indices {len(instruct_start_indices)} != number of instruction end indices {instruct_end_indices.sum()}"
            assert (len(instruct_start_indices[0]) == inputs_embeds.shape[0]), \
            f"number of instruction start indices {len(instruct_start_indices[0])} != number of LLM input batch size {inputs_embeds.shape[0]}"

            instruct_embeds = []
            instruct_seq_lens = []
            for i in range(inputs_embeds.shape[0]):
                start_index = instruct_start_indices[1][i]
                end_index = instruct_end_indices[1][i]
                instruct_embed = encoder_last_hidden_state[i, start_index+1:end_index]  # (L, C)
                instruct_embeds.append(instruct_embed)
                instruct_seq_lens.append(instruct_embed.shape[0])
            instruct_seq_lens = torch.tensor(instruct_seq_lens).to(inputs_llm.device)
            
            instruct_text_embeds = pad_tensors_wgrad(instruct_embeds)                   # (N, L, C)
            instruct_text_masks = gen_seq_masks(instruct_seq_lens)                      # (N, L)

            if self.output_thought:
                if self.model_type.startswith('flant5'):
                    outputs = self.Blip2InstructNav.t5_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=encoder_atts,
                        do_sample=self.use_nucleus_sampling,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        num_beams=self.num_beams,
                        max_new_tokens=self.max_length,
                        min_length=self.min_length,
                        repetition_penalty=self.repetition_penalty,
                        length_penalty=self.length_penalty,
                        num_return_sequences=self.num_captions,
                    )

                    output_text = self.Blip2InstructNav.t5_tokenizer.batch_decode(
                        outputs, skip_special_tokens=True
                    )
                else:
                    outputs = self.Blip2InstructNav.llm_model.generate(
                        inputs_embeds=inputs_embeds,
                        attention_mask=encoder_atts,
                        do_sample=self.use_nucleus_sampling,
                        top_p=self.top_p,
                        temperature=self.temperature,
                        num_beams=self.num_beams,
                        max_new_tokens=self.max_length,
                        min_length=self.min_length,
                        repetition_penalty=self.repetition_penalty,
                        length_penalty=self.length_penalty,
                        num_return_sequences=self.num_captions,
                    )

                    outputs[outputs == 0] = 2 # convert output id 0 to 2 (eos_token_id)
                    output_text = self.Blip2InstructNav.llm_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                    output_text = [text.strip() for text in output_text]

        return {
            "view_embeds": view_embeds,                               # (N * 32, C)
            "output_text": output_text,
            "instruct_text_embeds": instruct_text_embeds,             # (N, L, C)
            "instruct_text_masks": instruct_text_masks,               # (N, L)
            "loss": loss,
        }
    

class NavGPTAction(BertPreTrainedModel):
    def __init__(self, args):
        config = PretrainedConfig.from_pretrained('bert-base-uncased')
        # Now, iterate over the arguments and update the pre_config
        for arg in vars(args):
            value = getattr(args, arg)
            if value is not None:
                setattr(config, arg, value)
        super().__init__(config)

        self.gmap_pos_embeddings = nn.Sequential(
            nn.Linear(config.angle_feat_size + 3, config.hidden_size),
            BertLayerNorm(config.hidden_size, eps=1e-12)
        )
        self.gmap_step_embeddings = nn.Embedding(100, config.hidden_size)

        if config.global_cross_attn:
            self.global_cross_attn = True
            self.global_encoder = GACAEncoder(config)
        else:
            self.global_cross_attn = False
            self.global_encoder = GASAEncoder(config)
        
        if config.graph_sprels:
            self.sprel_linear = nn.Linear(1, 1)
        else:
            self.sprel_linear = None
        
        self.global_sap_head = ClsPrediction(config.hidden_size)

        # Whether to fuse local actions
        if config.fusion != 'global':
            self.local_encoder = GACAEncoder(config)
            self.local_sap_head = ClsPrediction(config.hidden_size)
            self.sap_fuse_linear = ClsPrediction(config.hidden_size, input_size=config.hidden_size*2)
        else:
            self.local_encoder = None

        self.init_weights()

    def _aggregate_gmap_features(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
    ):
        batch_size = len(split_traj_embeds)
        device = split_traj_embeds[0].device

        batch_gmap_img_fts = []
        for i in range(batch_size):
            visited_vp_fts, unvisited_vp_fts = {}, {}
            vp_masks = gen_seq_masks(split_traj_vp_lens[i])
            max_vp_len = max(split_traj_vp_lens[i])
            i_traj_embeds = split_traj_embeds[i][:, :max_vp_len] * vp_masks.unsqueeze(2)
            for t in range(len(split_traj_embeds[i])):
                visited_vp_fts[traj_vpids[i][t]] = torch.sum(i_traj_embeds[t], 0) / split_traj_vp_lens[i][t]
                for j, vp in enumerate(traj_cand_vpids[i][t]):
                    if vp not in visited_vp_fts:
                        unvisited_vp_fts.setdefault(vp, [])
                        unvisited_vp_fts[vp].append(i_traj_embeds[t][j])

            gmap_img_fts = []
            for vp in gmap_vpids[i][1:]:
                if vp in visited_vp_fts:
                    gmap_img_fts.append(visited_vp_fts[vp])
                else:
                    gmap_img_fts.append(torch.mean(torch.stack(unvisited_vp_fts[vp], 0), 0))
            gmap_img_fts = torch.stack(gmap_img_fts, 0)
            batch_gmap_img_fts.append(gmap_img_fts)

        batch_gmap_img_fts = pad_tensors_wgrad(batch_gmap_img_fts)
        # add a [stop] token at beginning
        batch_gmap_img_fts = torch.cat(
            [torch.zeros(batch_size, 1, batch_gmap_img_fts.size(2)).to(device), batch_gmap_img_fts], 
            dim=1
        )
        return batch_gmap_img_fts
    
    def gmap_input_embedding(
        self, split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens
    ):
        gmap_img_fts = self._aggregate_gmap_features(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids
        )
        gmap_embeds = gmap_img_fts + \
                      self.gmap_step_embeddings(gmap_step_ids) + \
                      self.gmap_pos_embeddings(gmap_pos_fts)
        gmap_masks = gen_seq_masks(gmap_lens)
        return gmap_embeds, gmap_masks

    def forward_sap(
        self,
        txt_embeds, txt_masks,
        split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
        gmap_step_ids, gmap_pos_fts, gmap_lens, graph_sprels=None
    ):

        gmap_embeds, gmap_masks = self.gmap_input_embedding(
            split_traj_embeds, split_traj_vp_lens, traj_vpids, traj_cand_vpids, gmap_vpids,
            gmap_step_ids, gmap_pos_fts, gmap_lens
        )
        
        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(graph_sprels.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None

        if self.global_cross_attn:
            gmap_embeds = self.global_encoder(
                txt_embeds, txt_masks,
                gmap_embeds, gmap_masks,
                graph_sprels=graph_sprels
            )
        else:
            gmap_embeds = self.global_encoder(
                gmap_embeds, gmap_masks,
                graph_sprels=graph_sprels
            )
        return gmap_embeds
       
    def forward_per_step(
        self,
        txt_embeds, txt_masks,
        gmap_img_embeds, gmap_step_ids, gmap_pos_fts, 
        gmap_masks, gmap_pair_dists, gmap_visited_masks, gmap_vpids,
        pano_embeds, pano_masks, vp_cand_vpids
    ):
        batch_size = txt_embeds.size(0)

        # global branch
        gmap_embeds = gmap_img_embeds + \
                      self.gmap_step_embeddings(gmap_step_ids) + \
                      self.gmap_pos_embeddings(gmap_pos_fts)
        
        if self.sprel_linear is not None:
            graph_sprels = self.sprel_linear(
                gmap_pair_dists.unsqueeze(3)).squeeze(3).unsqueeze(1)
        else:
            graph_sprels = None
        
        if self.global_cross_attn:
            gmap_embeds = self.global_encoder(
                txt_embeds, txt_masks,
                gmap_embeds, gmap_masks,
                graph_sprels=graph_sprels
            )
        else:
            gmap_embeds = self.global_encoder(
                gmap_embeds, gmap_masks,
                graph_sprels=graph_sprels
            )

        if self.local_encoder:

            # local branch
            vp_embeds = self.local_encoder(txt_embeds, txt_masks, pano_embeds, pano_masks)

            # navigation logits
            fuse_weights = torch.sigmoid(self.sap_fuse_linear(
                torch.cat([gmap_embeds[:, 0], vp_embeds[:, 0]], 1)
            ))

            global_logits = self.global_sap_head(gmap_embeds).squeeze(2) * fuse_weights
            global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
            global_logits.masked_fill_(gmap_masks.logical_not(), -float('inf'))

            local_logits = self.local_sap_head(vp_embeds).squeeze(2) * (1 - fuse_weights)
            local_logits.masked_fill_(pano_masks.logical_not(), -float('inf'))

            # fusion
            fused_logits = torch.clone(global_logits)
            fused_logits[:, 0] += local_logits[:, 0]   # stop
            for i in range(batch_size):
                visited_nodes = set([vp for vp, mask in zip(gmap_vpids[i], gmap_visited_masks[i]) if mask])
                tmp = {}
                bw_logits = 0
                for j, cand_vpid in enumerate(vp_cand_vpids[i]):
                    if j > 0:
                        if cand_vpid in visited_nodes:
                            bw_logits += local_logits[i, j]
                        else:
                            tmp[cand_vpid] = local_logits[i, j]
                for j, vp in enumerate(gmap_vpids[i]):
                    if j > 0 and vp not in visited_nodes:
                        if vp in tmp:
                            fused_logits[i, j] += tmp[vp]
                        else:
                            fused_logits[i, j] += bw_logits
        else:
            global_logits = self.global_sap_head(gmap_embeds).squeeze(2)
            global_logits.masked_fill_(gmap_visited_masks, -float('inf'))
            global_logits.masked_fill_(gmap_masks.logical_not(), -float('inf'))
            vp_embeds = None
            local_logits = None
            fused_logits = None

        outs = {
            'gmap_embeds': gmap_embeds,
            'vp_embeds': vp_embeds,
            'global_logits': global_logits,
            'local_logits': local_logits,
            'fused_logits': fused_logits,
        }
        return outs


    def forward(self, mode, batch, **kwargs):
        # forward sap in pretrain mode
        if mode == 'sap':
            return self.forward_sap(
                batch['text_embeds'], batch['text_masks'],
                batch['split_traj_embeds'], batch['split_traj_vp_lens'], batch['traj_vpids'], 
                batch['traj_cand_vpids'], batch['gmap_vpids'], batch['gmap_step_ids'], 
                batch['gmap_pos_fts'], batch['gmap_lens'], batch['graph_sprels']
            )
        # forward in inference or finetune mode
        elif mode == 'per_step':
            return self.forward_per_step(
                batch['text_embeds'], batch['text_masks'],
                batch['gmap_img_embeds'], batch['gmap_step_ids'], batch['gmap_pos_fts'], 
                batch['gmap_masks'], batch['gmap_pair_dists'], batch['gmap_visited_masks'], batch['gmap_vpids'],
                batch['vp_img_embeds'], batch['vp_masks'], batch['vp_cand_vpids']
            )
       

class NavGPT(nn.Module):
    def __init__(self, config):
        super().__init__()

        if config.single_action_head:
            print("[INFO] Using single action head, setting global_cross_attn to False, num_x_layers to 1, and num_pano_layers to 0.")
            config.global_cross_attn = False
            config.num_x_layers = 1
            config.num_pano_layers = 0
        
        self.img_embeddings = ImageEmbeddings(config)
        self.llm = NavGPTThought(config)
        self.policy = NavGPTAction(config)

    def forward(self, mode, batch, **kwargs):
        batch = collections.defaultdict(lambda: None, batch)

        if mode == 'thought':
            return self.forward_thought_per_step(batch)

        elif mode == 'action':
            return self.forward_action_per_step(batch)

        elif mode == "panorama":
            return self.forward_panorama_per_step(batch)
    
    def forward_thought_per_step(self, batch, **kwargs):
        
        local_outputs = self.llm(
            batch["view_img_fts"],
            batch["loc_fts"],
            batch["qformer_text_inputs"],
            batch["text_inputs"],
            # batch["text_outputs"],
            None,
        )
        return local_outputs
    
    def forward_action_per_step(self, batch, **kwargs):
        
        # forward global encoder
        nav_outputs = self.policy(
            mode = 'per_step',
            batch = batch,
        )
        return nav_outputs
    
    def forward_panorama_per_step(self, batch, **kwargs):

        # forward image embeddings
        img_embeds = self.img_embeddings(
            mode = 'per_step',
            batch = batch,
        )
        return img_embeds



class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()
        self.state2value = nn.Sequential(
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            nn.Linear(512, 1),
        )

    def forward(self, state):
        return self.state2value(state).squeeze()