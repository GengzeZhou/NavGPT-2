TOKENIZERS_PARALLELISM="false" python -m torch.distributed.run --nproc_per_node=1 train.py --cfg-path lavis/projects/blip2/train/r2r_NavGPT_ft_vicuna13b.yaml