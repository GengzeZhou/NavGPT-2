import json
import os
import sys
import numpy as np
import random
import math
import time
from collections import defaultdict

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from utils.distributed import is_default_gpu
from utils.ops import pad_tensors, gen_seq_masks
from torch.nn.utils.rnn import pad_sequence

from .agent_base import Seq2SeqAgent
from .eval_utils import cal_dtw

from models.graph_utils import GraphMap
from models.NavGPT_model import NavGPT, Critic
from models.ops import pad_tensors_wgrad

from .prompt_template import NavGPT_PROMPT

from transformers import PretrainedConfig

class GMapNavAgent(Seq2SeqAgent):
    
    def _build_model(self, config):
        ''' Build model, either from scratch or from saved model checkpoint. '''
        self.NavGPT = NavGPT(config).to(self.device)
        if config.bert_ckpt_file is not None:
            ckpt_weights = torch.load(config.bert_ckpt_file)['NavGPT']['state_dict']
            self.NavGPT.load_state_dict(ckpt_weights, strict=False)
        
        if config.freeze_qformer:
            print("[INFO] Freezing the Q-Former.")
            for name, param in self.NavGPT.llm.Blip2InstructNav.named_parameters():
                param.requires_grad = False
        
        self.critic = Critic(self.args).to(self.device)
        # prompt
        self.prompt = NavGPT_PROMPT
        # buffer
        self.scanvp_cands = {}

    def _construct_candidate_dict(self, rel_angles, rel_dists):
        ''' Construct candidate dict. '''
        image_p = '[IMG]<image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image><image>[/IMG]'

        candidate_dict = {}
        for i in range(len(rel_angles)):
            heading = np.rad2deg(rel_angles[i, 0])
            if -45 <= heading <= 45:
                direction = "front"
            elif 45 < heading <= 135:
                direction = "right"
            elif -135 <= heading < -45:
                direction = "left"
            else:
                direction = "rear"
            key = f"Candidate {i}, facing {heading:.2f} degrees, {direction}"
            # key = f"Candidate {i}, facing {np.rad2deg(rel_angles[i, 0]):.2f} degrees, {rel_dists[i]:.2f} meters"
            candidate_dict[key] = image_p
        return candidate_dict

    def _language_variable(self, instructions, batch_view_lens, batch_rel_angles, batch_rel_dists):
        ''' Construct language variable. '''
        batch_qformer_text_inputs, batch_text_inputs = [], []
        for i, l in enumerate(batch_view_lens):
            batch_qformer_text_inputs += [instructions[i]] * l

        for i in range(len(batch_view_lens)):
            prompt = self.prompt.replace("{instruction}", instructions[i])
            candidate_dict = self._construct_candidate_dict(batch_rel_angles[i], batch_rel_dists[i])
            prompt = prompt.replace("{candidate}", str(candidate_dict))
            batch_text_inputs.append(prompt)
            
        return batch_qformer_text_inputs, batch_text_inputs

    def _local_feature_variable(self, obs, gmaps, instructions):
        ''' Extract precomputed features into variable. '''
        batch_view_img_fts, batch_view_cls_fts, batch_loc_fts = [], [], []
        batch_view_lens, batch_cand_vpids = [], []
        batch_rel_angles, batch_rel_dists = [], []
        
        for i, ob in enumerate(obs):
            view_img_fts, cand_vpids = [], []
            # cand views
            used_viewidxs = set()
            for j, cc in enumerate(ob['candidate']):
                view_img_fts.append(cc['feature'])            # (257, 1024) or (3, 224, 224)
                cand_vpids.append(cc['viewpointId'])
                used_viewidxs.add(cc['pointId'])

            cur_rel_angles, cur_rel_dists, cur_cand_pos_fts = gmaps[i].get_pos_fts(       # (n_candidates, 2), (n_candidates, 3), (n_candidates, 7)
                obs[i]['viewpoint'], cand_vpids, 
                obs[i]['heading'], obs[i]['elevation']
            )
            _, _, cur_start_pos_fts = gmaps[i].get_pos_fts(   # (1, 7)
                obs[i]['viewpoint'], [gmaps[i].start_vp], 
                obs[i]['heading'], obs[i]['elevation']
            )

            # combine cand views and noncand views
            view_img_fts = np.stack(view_img_fts, 0)          # (n_candidates, 257, 1024) or (n_candidates, 3, 224, 224)
            view_img_cls_fts = view_img_fts[:, 0]             # (n_candidates, 1024)
            vp_loc_fts = np.zeros((len(view_img_fts), 14), dtype=np.float32)
            vp_loc_fts[:, :7] = cur_start_pos_fts
            vp_loc_fts[:, 7:] = cur_cand_pos_fts

            batch_view_img_fts.append(torch.from_numpy(view_img_fts))
            batch_view_cls_fts.append(torch.from_numpy(view_img_cls_fts))
            batch_loc_fts.append(torch.from_numpy(vp_loc_fts))
            batch_cand_vpids.append(cand_vpids)
            batch_view_lens.append(len(view_img_fts))
            batch_rel_angles.append(cur_rel_angles)
            batch_rel_dists.append(cur_rel_dists[:, 0] * 30)  # distances are normalized by 30m

        # pad features to max_len
        batch_view_img_fts = torch.cat(batch_view_img_fts).to(self.device)
        batch_view_cls_fts = pad_tensors(batch_view_cls_fts).to(self.device)
        batch_loc_fts = torch.cat(batch_loc_fts).to(self.device)
        batch_view_lens = torch.LongTensor(batch_view_lens).to(self.device)
        batch_qformer_text_inputs, batch_text_inputs = self._language_variable(instructions, batch_view_lens, batch_rel_angles, batch_rel_dists)

        return {
            'view_cls_fts': batch_view_cls_fts,
            'view_img_fts': batch_view_img_fts, 'loc_fts': batch_loc_fts,
            'view_lens': batch_view_lens, 'cand_vpids': batch_cand_vpids,
            'qformer_text_inputs': batch_qformer_text_inputs,
            'text_inputs': batch_text_inputs,
        }

    def _nav_vp_variable(self, pano_embeds, cand_vpids, view_lens):

        # add [stop] token
        vp_img_embeds = torch.cat(
            [torch.zeros_like(pano_embeds[:, :1]), pano_embeds], 1
        )

        return {
            'vp_img_embeds': vp_img_embeds,
            'vp_masks': gen_seq_masks(view_lens+1),
            'vp_cand_vpids': [[None]+x for x in cand_vpids],
        }

    def _nav_gmap_variable(self, obs, gmaps):
        # [stop] + gmap_vpids
        batch_size = len(obs)
        
        batch_gmap_vpids, batch_gmap_lens = [], []
        batch_gmap_img_embeds, batch_gmap_step_ids, batch_gmap_pos_fts = [], [], []
        batch_gmap_pair_dists, batch_gmap_visited_masks = [], []
        batch_no_vp_left = []
        for i, gmap in enumerate(gmaps):
            visited_vpids, unvisited_vpids = [], []                
            for k in gmap.node_positions.keys():
                if self.args.act_visited_nodes:
                    if k == obs[i]['viewpoint']:
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
                else:
                    if gmap.graph.visited(k):
                        visited_vpids.append(k)
                    else:
                        unvisited_vpids.append(k)
            batch_no_vp_left.append(len(unvisited_vpids) == 0)
            if self.args.enc_full_graph:
                gmap_vpids = [None] + visited_vpids + unvisited_vpids
                gmap_visited_masks = [0] + [1] * len(visited_vpids) + [0] * len(unvisited_vpids)
            else:
                gmap_vpids = [None] + unvisited_vpids
                gmap_visited_masks = [0] * len(gmap_vpids)

            gmap_step_ids = [gmap.node_step_ids.get(vp, 0) for vp in gmap_vpids]
            gmap_img_embeds = [gmap.get_node_embed(vp) for vp in gmap_vpids[1:]]
            gmap_img_embeds = torch.stack(
                [torch.zeros_like(gmap_img_embeds[0])] + gmap_img_embeds, 0
            )   # cuda

            _, _, gmap_pos_fts = gmap.get_pos_fts(
                obs[i]['viewpoint'], gmap_vpids, obs[i]['heading'], obs[i]['elevation'],
            )

            gmap_pair_dists = np.zeros((len(gmap_vpids), len(gmap_vpids)), dtype=np.float32)
            for i in range(1, len(gmap_vpids)):
                for j in range(i+1, len(gmap_vpids)):
                    gmap_pair_dists[i, j] = gmap_pair_dists[j, i] = \
                        gmap.graph.distance(gmap_vpids[i], gmap_vpids[j])

            batch_gmap_img_embeds.append(gmap_img_embeds)
            batch_gmap_step_ids.append(torch.LongTensor(gmap_step_ids))
            batch_gmap_pos_fts.append(torch.from_numpy(gmap_pos_fts))
            batch_gmap_pair_dists.append(torch.from_numpy(gmap_pair_dists))
            batch_gmap_visited_masks.append(torch.BoolTensor(gmap_visited_masks))
            batch_gmap_vpids.append(gmap_vpids)
            batch_gmap_lens.append(len(gmap_vpids))

        # collate
        batch_gmap_lens = torch.LongTensor(batch_gmap_lens)
        batch_gmap_masks = gen_seq_masks(batch_gmap_lens).to(self.device)
        batch_gmap_img_embeds = pad_tensors_wgrad(batch_gmap_img_embeds)
        batch_gmap_step_ids = pad_sequence(batch_gmap_step_ids, batch_first=True).to(self.device)
        batch_gmap_pos_fts = pad_tensors(batch_gmap_pos_fts).to(self.device)
        batch_gmap_visited_masks = pad_sequence(batch_gmap_visited_masks, batch_first=True).to(self.device)

        max_gmap_len = max(batch_gmap_lens)
        gmap_pair_dists = torch.zeros(batch_size, max_gmap_len, max_gmap_len).float()
        for i in range(batch_size):
            gmap_pair_dists[i, :batch_gmap_lens[i], :batch_gmap_lens[i]] = batch_gmap_pair_dists[i]
        gmap_pair_dists = gmap_pair_dists.to(self.device)

        return {
            'gmap_vpids': batch_gmap_vpids, 'gmap_img_embeds': batch_gmap_img_embeds, 
            'gmap_step_ids': batch_gmap_step_ids, 'gmap_pos_fts': batch_gmap_pos_fts,
            'gmap_visited_masks': batch_gmap_visited_masks, 
            'gmap_pair_dists': gmap_pair_dists, 'gmap_masks': batch_gmap_masks,
            'no_vp_left': batch_no_vp_left,
        }

    def _teacher_action(self, obs, vpids, ended, visited_masks=None):
        """
        Extract teacher actions into variable.
        :param obs: The observation.
        :param ended: Whether the action seq is ended
        :return:
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if ob['viewpoint'] == ob['gt_path'][-1]:
                    a[i] = 0    # Stop if arrived 
                else:
                    scan = ob['scan']
                    cur_vp = ob['viewpoint']
                    min_idx, min_dist = self.args.ignoreid, float('inf')
                    for j, vpid in enumerate(vpids[i]):
                        if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                            # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                            dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                    + self.env.shortest_distances[scan][cur_vp][vpid]
                            if dist < min_dist:
                                min_dist = dist
                                min_idx = j
                    a[i] = min_idx
                    if min_idx == self.args.ignoreid:
                        print('scan %s: all vps are searched' % (scan))

        return torch.from_numpy(a).to(self.device)

    def _teacher_action_r4r(
        self, obs, vpids, ended, visited_masks=None, imitation_learning=False, t=None, traj=None
    ):
        """R4R is not the shortest path. The goal location can be visited nodes.
        """
        a = np.zeros(len(obs), dtype=np.int64)
        for i, ob in enumerate(obs):
            if ended[i]:                                            # Just ignore this index
                a[i] = self.args.ignoreid
            else:
                if imitation_learning:
                    assert ob['viewpoint'] == ob['gt_path'][t]
                    if t == len(ob['gt_path']) - 1:
                        a[i] = 0    # stop
                    else:
                        goal_vp = ob['gt_path'][t + 1]
                        for j, vpid in enumerate(vpids[i]):
                            if goal_vp == vpid:
                                a[i] = j
                                break
                else:
                    if ob['viewpoint'] == ob['gt_path'][-1]:
                        a[i] = 0    # Stop if arrived 
                    else:
                        scan = ob['scan']
                        cur_vp = ob['viewpoint']
                        min_idx, min_dist = self.args.ignoreid, float('inf')
                        for j, vpid in enumerate(vpids[i]):
                            if j > 0 and ((visited_masks is None) or (not visited_masks[i][j])):
                                if self.args.expert_policy == 'ndtw':
                                    dist = - cal_dtw(
                                        self.env.shortest_distances[scan], 
                                        sum(traj[i]['path'], []) + self.env.shortest_paths[scan][ob['viewpoint']][vpid][1:], 
                                        ob['gt_path'], 
                                        threshold=3.0
                                    )['nDTW']
                                elif self.args.expert_policy == 'spl':
                                    # dist = min([self.env.shortest_distances[scan][vpid][end_vp] for end_vp in ob['gt_end_vps']])
                                    dist = self.env.shortest_distances[scan][vpid][ob['gt_path'][-1]] \
                                            + self.env.shortest_distances[scan][cur_vp][vpid]
                                if dist < min_dist:
                                    min_dist = dist
                                    min_idx = j
                        a[i] = min_idx
                        if min_idx == self.args.ignoreid:
                            print('scan %s: all vps are searched' % (scan))
        return torch.from_numpy(a).to(self.device)

    def make_equiv_action(self, a_t, gmaps, obs, traj=None):
        """
        Interface between Panoramic view and Egocentric view
        It will convert the action panoramic view action a_t to equivalent egocentric view actions for the simulator
        """
        for i, ob in enumerate(obs):
            action = a_t[i]
            if action is not None:            # None is the <stop> action
                traj[i]['path'].append(gmaps[i].graph.path(ob['viewpoint'], action))
                if len(traj[i]['path'][-1]) == 1:
                    prev_vp = traj[i]['path'][-2][-1]
                else:
                    prev_vp = traj[i]['path'][-1][-2]
                viewidx = self.scanvp_cands['%s_%s'%(ob['scan'], prev_vp)][action]
                heading = (viewidx % 12) * math.radians(30)
                elevation = (viewidx // 12 - 1) * math.radians(30)
                self.env.env.sims[i].newEpisode([ob['scan']], [action], [heading], [elevation])

    def _update_scanvp_cands(self, obs):
        for ob in obs:
            scan = ob['scan']
            vp = ob['viewpoint']
            scanvp = '%s_%s' % (scan, vp)
            self.scanvp_cands.setdefault(scanvp, {})
            for cand in ob['candidate']:
                self.scanvp_cands[scanvp].setdefault(cand['viewpointId'], {})
                self.scanvp_cands[scanvp][cand['viewpointId']] = cand['pointId']

    # @profile
    def rollout(self, train_ml=None, train_rl=False, reset=True):
        if reset:  # Reset env
            obs = self.env.reset()
        else:
            obs = self.env._get_obs()
        self._update_scanvp_cands(obs)

        batch_size = len(obs)
        # build graph: keep the start viewpoint
        gmaps = [GraphMap(ob['viewpoint']) for ob in obs]
        for i, ob in enumerate(obs):
            gmaps[i].update_graph(ob)

        # Record the navigation path
        traj = [{
            'instr_id': ob['instr_id'],
            'path': [[ob['viewpoint']]],
            'details': {},
            'thoughts': [],
        } for ob in obs]

        # Language inputs
        instructions = [ob['instruction'] for ob in obs]
    
        # Initialization the tracking state
        ended = np.array([False] * batch_size)
        just_ended = np.array([False] * batch_size)

        # Init the logs
        masks = []
        entropys = []
        IL_loss = 0.   
        acc_IL_loss = 0.
        acc_g_loss = 0.  

        for t in range(self.args.max_action_steps):

            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    gmap.node_step_ids[obs[i]['viewpoint']] = t + 1
                # for k, v in gmap.node_embeds.items():
                #     gmap.node_embeds[k][0] = v[0].detach()

            # graph representation
            local_inputs = self._local_feature_variable(obs, gmaps, instructions)

            # forward NavGPT thoughts
            local_outputs = self.NavGPT('thought', local_inputs)
            view_embeds, instruct_text_embeds, instruct_text_masks = local_outputs["view_embeds"], local_outputs["instruct_text_embeds"], local_outputs["instruct_text_masks"]
            thoughts, generation_loss = local_outputs["output_text"], local_outputs["loss"]
            local_inputs['text_embeds'] = instruct_text_embeds
            local_inputs['text_masks'] = instruct_text_masks
            local_inputs['view_llm_fts'] = view_embeds

            # split loc_fts
            split_loc_fts = torch.split(local_inputs['loc_fts'], local_inputs['view_lens'].tolist(), 0)
            local_inputs['loc_fts'] = pad_tensors_wgrad(split_loc_fts)


            # Get node embeddings
            pano_embeds, pano_masks = self.NavGPT('panorama', local_inputs)

            # Use the average of the view_embeds as the visited node embedding
            avg_pano_embeds = torch.sum(pano_embeds * pano_masks.unsqueeze(2), 1) / \
                        torch.sum(pano_masks, 1, keepdim=True)

            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    # update visited node
                    i_vp = obs[i]['viewpoint']
                    gmap.update_node_embed(i_vp, avg_pano_embeds[i], rewrite=True)
                    # update unvisited nodes
                    for j, i_cand_vp in enumerate(local_inputs['cand_vpids'][i]):
                        if not gmap.graph.visited(i_cand_vp):
                            gmap.update_node_embed(i_cand_vp, pano_embeds[i, j])

            # navigation policy
            nav_inputs = self._nav_gmap_variable(obs, gmaps)
            nav_inputs.update(local_inputs)

            # add [stop] token for pano embeddings
            nav_vp_inputs = self._nav_vp_variable(pano_embeds, local_inputs['cand_vpids'], local_inputs['view_lens'])
            nav_inputs.update(nav_vp_inputs)

            nav_outs = self.NavGPT('action', nav_inputs)

            if self.args.fusion == 'local':
                nav_logits = nav_outs['local_logits']
                nav_vpids = nav_inputs['vp_cand_vpids']
            elif self.args.fusion == 'global':
                nav_logits = nav_outs['global_logits']
                nav_vpids = nav_inputs['gmap_vpids']
            else:
                nav_logits = nav_outs['fused_logits']
                nav_vpids = nav_inputs['gmap_vpids']

            nav_probs = torch.softmax(nav_logits, 1)
            
            # update graph
            for i, gmap in enumerate(gmaps):
                if not ended[i]:
                    i_vp = obs[i]['viewpoint']
                    gmap.node_stop_scores[i_vp] = {
                        'stop': nav_probs[i, 0].data.item(),
                    }
                                        
            if train_ml is not None:
                # Supervised training
                nav_targets = self._teacher_action_r4r(
                    obs, nav_vpids, ended, 
                    visited_masks=nav_inputs['gmap_visited_masks'] if self.args.fusion != 'local' else None,
                    imitation_learning=(self.feedback=='teacher'), t=t, traj=traj
                )

                IL_loss = self.criterion(nav_logits, nav_targets)
                IL_loss = IL_loss * train_ml / (batch_size * self.args.accumulate_grad_step)
                acc_IL_loss += IL_loss
                step_loss = IL_loss

                if generation_loss is not None:
                    g_loss = generation_loss * train_ml / (batch_size * self.args.accumulate_grad_step)
                    acc_g_loss += g_loss
                    step_loss += g_loss
                
                if self.args.step_update:
                    step_loss.backward()
                                                 
            # Determinate the next navigation viewpoint
            if self.feedback == 'teacher':
                a_t = nav_targets                 # teacher forcing
            elif self.feedback == 'argmax':
                _, a_t = nav_logits.max(1)        # student forcing - argmax
                a_t = a_t.detach() 
            elif self.feedback == 'sample':
                c = torch.distributions.Categorical(nav_probs)
                self.logs['entropy'].append(c.entropy().sum().item())            # For log
                entropys.append(c.entropy())                                     # For optimization
                a_t = c.sample().detach() 
            elif self.feedback == 'expl_sample':
                _, a_t = nav_probs.max(1)
                rand_explores = np.random.rand(batch_size, ) > self.args.expl_max_ratio  # hyper-param
                if self.args.fusion == 'local':
                    cpu_nav_masks = nav_inputs['vp_nav_masks'].data.cpu().numpy()
                else:
                    cpu_nav_masks = (nav_inputs['gmap_masks'] * nav_inputs['gmap_visited_masks'].logical_not()).data.cpu().numpy()
                for i in range(batch_size):
                    if rand_explores[i]:
                        cand_a_t = np.arange(len(cpu_nav_masks[i]))[cpu_nav_masks[i]]
                        a_t[i] = np.random.choice(cand_a_t)
            else:
                print(self.feedback)
                sys.exit('Invalid feedback option')

            # Determine stop actions
            if self.feedback == 'teacher' or self.feedback == 'sample': # in training
                # a_t_stop = [ob['viewpoint'] in ob['gt_end_vps'] for ob in obs]
                a_t_stop = [ob['viewpoint'] == ob['gt_path'][-1] for ob in obs]
            else:
                a_t_stop = a_t == 0

            # Prepare environment action
            cpu_a_t = []  
            for i in range(batch_size):
                if a_t_stop[i] or ended[i] or nav_inputs['no_vp_left'][i] or (t == self.args.max_action_steps - 1):
                    cpu_a_t.append(None)
                    just_ended[i] = True
                else:
                    cpu_a_t.append(nav_vpids[i][a_t[i]])   

            # Make action and get the new state
            self.make_equiv_action(cpu_a_t, gmaps, obs, traj)
            for i in range(batch_size):
                if (not ended[i]) and just_ended[i]:
                    stop_node, stop_score = None, {'stop': -float('inf')}
                    for k, v in gmaps[i].node_stop_scores.items():
                        if v['stop'] > stop_score['stop']:
                            stop_score = v
                            stop_node = k
                    if stop_node is not None and obs[i]['viewpoint'] != stop_node:
                        traj[i]['path'].append(gmaps[i].graph.path(obs[i]['viewpoint'], stop_node))
                    if self.args.detailed_output:
                        for k, v in gmaps[i].node_stop_scores.items():
                            traj[i]['details'][k] = {
                                'stop_prob': float(v['stop']),
                            }
                if self.args.output_thought:
                    traj[i]['thoughts'].append(thoughts[i])

            # new observation and update graph
            obs = self.env._get_obs()
            self._update_scanvp_cands(obs)
            for i, ob in enumerate(obs):
                if not ended[i]:
                    gmaps[i].update_graph(ob)

            ended[:] = np.logical_or(ended, np.array([x is None for x in cpu_a_t]))

            # Early exit if all ended
            if ended.all():
                break

        if train_ml is not None:
            if generation_loss is not None:
                self.logs['generation_loss'].append(acc_g_loss.item())
            self.logs['IL_loss'].append(acc_IL_loss.item())
            # Loss for the whole trajectory
            self.loss = acc_IL_loss + acc_g_loss

        return traj
