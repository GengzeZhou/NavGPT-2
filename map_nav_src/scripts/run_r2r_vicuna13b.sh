DATA_ROOT=../datasets

train_alg=dagger

features=eva-clip-g
ft_dim=1408

ngpus=1
seed=0
batch_size=2

name=NavGPT2-Vicuna13b
name=${name}-seed.${seed}
name=${name}-bs${batch_size}

outdir=${DATA_ROOT}/R2R/exprs_map/finetune/${name}

flag="--root_dir ${DATA_ROOT}
      --dataset r2r
      --output_dir ${outdir}
      --world_size ${ngpus}
      --seed ${seed}
      --tokenizer bert      

      --enc_full_graph
      --graph_sprels
      --fusion global

      --expert_policy spl
      --train_alg ${train_alg}
      
      --num_x_layers 4
      
      --max_action_steps 15
      --max_instr_len 200

      --batch_size ${batch_size}
      --lr 1e-5
      --iters 200000
      --log_every 2500
      --optim adamW

      --features ${features}
      --image_feat_size ${ft_dim}
      --angle_feat_size 4

      --ml_weight 0.2   

      --feat_dropout 0.4
      --dropout 0.5
      
      --gamma 0."

# train
# replace "--qformer_ckpt_path" with the path to the pretrained qformer
CUDA_VISIBLE_DEVICES='0' python r2r/main_nav.py $flag  \
        --freeze_qformer \
        --step_update \
        --arch blip2_vicuna_instruct_nav \
        --model_type vicuna13b \
        --llm_ctx_size 4096 \
        --qformer_ckpt_path path to the pretrained qformer \
        --aug ../datasets/R2R/annotations/prevalent_aug.json \