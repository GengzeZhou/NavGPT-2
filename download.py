import argparse
from huggingface_hub import snapshot_download

def main(args):
    if args.data:
        if args.dataset == "r2r" or args.dataset == "all":
            # Download the R2R dataset
            snapshot_download(repo_id="ZGZzz/NavGPT-R2R", repo_type="dataset", allow_patterns="*.zip.*", local_dir="datasets", local_dir_use_symlinks=False)

        if args.dataset == "instruct" or args.dataset == "all":
            snapshot_download(repo_id="ZGZzz/NavGPT-Instruct", repo_type="dataset", allow_patterns="*.json", local_dir="datasets/NavGPT-Instruct", local_dir_use_symlinks=False)
    
    if args.checkpoints:
        if args.model == "xl" or args.model == "all":
            # Download the NavGPT-2 policy model
            snapshot_download(repo_id="ZGZzz/NavGPT2-FlanT5-XL", repo_type="model", allow_patterns="best_val_unseen_xl", local_dir="datasets/R2R/trained_models", local_dir_use_symlinks=False)

            # Download the NavGPT-2 pretrained Q-former
            snapshot_download(repo_id="ZGZzz/NavGPT2-FlanT5-XL", repo_type="model", allow_patterns="*.pth", local_dir="map_nav_src/models/lavis/output", local_dir_use_symlinks=False)
        
        if args.model == "xxl" or args.model == "all":
            # Download the NavGPT-2 policy model
            snapshot_download(repo_id="ZGZzz/NavGPT2-FlanT5-XXL", repo_type="model", allow_patterns="best_val_unseen_xxl", local_dir="datasets/R2R/trained_models", local_dir_use_symlinks=False)

            # Download the NavGPT-2 pretrained Q-former
            snapshot_download(repo_id="ZGZzz/NavGPT2-FlanT5-XXL", repo_type="model", allow_patterns="*.pth", local_dir="map_nav_src/models/lavis/output", local_dir_use_symlinks=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", action="store_true", default=False, help="Download the R2R dataset and instruction tuning data for NavGPT-2")
    parser.add_argument("--dataset", type=str, default="all", choices=["r2r", "instruct", "all"], help="Dataset to download")
    parser.add_argument("--checkpoints", action="store_true", default=False, help="Download the NavGPT-2 policy model and pretrained Q-former")
    parser.add_argument("--model", type=str, default="all", choices=["xl", "xxl", "all"], help="Model type to download")
    args = parser.parse_args()
    main(args)