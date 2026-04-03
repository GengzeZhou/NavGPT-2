NUM_GPUS=${NUM_GPUS:-$(python -c "import torch; print(torch.cuda.device_count())")}
TOKENIZERS_PARALLELISM="false" python -m torch.distributed.run --nproc_per_node=$NUM_GPUS train.py --cfg-path lavis/projects/blip2/train/r2r_NavGPT_ft_vicuna13b.yaml
