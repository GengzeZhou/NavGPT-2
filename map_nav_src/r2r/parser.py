import argparse
import os


def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--root_dir', type=str, default='../datasets')
    parser.add_argument('--dataset', type=str, default='r2r', choices=['r2r', 'r4r', 'rxr-en', 'REVERIE'])
    parser.add_argument('--output_dir', type=str, default='../datasets/R2R/exprs_map/finetune/default', help='experiment id')
    parser.add_argument('--seed', type=int, default=0)

    parser.add_argument('--tokenizer', choices=['bert', 'xlm'], default='bert')

    parser.add_argument('--act_visited_nodes', action='store_true', default=False)
    parser.add_argument('--fusion', choices=['global', 'local', 'dynamic'], default='global')
    parser.add_argument('--expl_sample', action='store_true', default=False)
    parser.add_argument('--expl_max_ratio', type=float, default=0.6)
    parser.add_argument('--expert_policy', default='spl', choices=['spl', 'ndtw'])

    # distributional training (single-node, multiple-gpus)
    parser.add_argument('--world_size', type=int, default=1, help='number of gpus')
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument("--node_rank", type=int, default=0, help="Id of the node")
    
    # General
    parser.add_argument('--iters', type=int, default=200000, help='training iterations')
    parser.add_argument('--accumulate_grad_step', type=int, default=1, help='accumulate gradient step')
    parser.add_argument('--log_every', type=int, default=1000)
    parser.add_argument('--eval_first', action='store_true', default=False)
    parser.add_argument('--step_update', action='store_true', default=False)

    # Data preparation
    parser.add_argument('--max_instr_len', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--ignoreid', type=int, default=-100, help='ignoreid for action')
    
    # Load the model from
    parser.add_argument("--resume_file", default=None, help='path of the trained model')
    parser.add_argument("--resume_optimizer", action="store_true", default=False)

    # Augmented Paths from
    parser.add_argument("--aug", default=None)
    parser.add_argument('--bert_ckpt_file', default=None, help='init vlnbert')

    # Listener Model Config
    parser.add_argument("--ml_weight", type=float, default=0.20)
    parser.add_argument('--entropy_loss_weight', type=float, default=0.01)

    parser.add_argument("--features", type=str, default='eva-clip-g')
    
    # VLM Model Config
    parser.add_argument('--arch', choices=['blip2_t5_instruct_nav', 'blip2_vicuna_instruct_nav'], default='blip2_t5_instruct_nav')
    parser.add_argument('--model_type', choices=['flant5xl', 'flant5xxl', 'vicuna7b', 'vicuna13b'], default='flant5xl')
    # parser.add_argument('--qformer_ckpt_path', type=str, default="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/InstructBLIP/instruct_blip_flanxl_trimmed.pth")
    parser.add_argument('--qformer_ckpt_path', type=str, default=None)
    parser.add_argument('--qformer_pos_emb', action='store_true', default=False)
    parser.add_argument('--load_patch_feature', action='store_true', default=True)
    parser.add_argument('--output_thought', action='store_true', default=False)
    parser.add_argument('--freeze_qformer', action='store_true', default=True)
    # Generation config
    parser.add_argument('--use_nucleus_sampling', action='store_true', default=False)
    parser.add_argument('--num_beams', type=int, default=5)
    parser.add_argument('--max_length', type=int, default=256)
    parser.add_argument('--min_length', type=int, default=1)
    parser.add_argument('--repetition_penalty', type=float, default=1.5)
    parser.add_argument('--length_penalty', type=float, default=1.0)
    parser.add_argument('--num_captions', type=int, default=1)
    parser.add_argument('--top_p', type=float, default=0.9)
    parser.add_argument('--temperature', type=float, default=1.0)

    # Global branch config
    parser.add_argument('--num_x_layers', type=int, default=4)
    parser.add_argument('--num_pano_layers', type=int, default=2)
    parser.add_argument('--num_attention_heads', type=int, default=12)
    parser.add_argument('--attention_probs_dropout_prob', type=int, default=0.1)
    parser.add_argument('--hidden_dropout_prob', type=int, default=0.1)
    parser.add_argument('--layer_norm_eps', type=int, default=1e-12)
    parser.add_argument('--hidden_size', type=int, default=768, help='DuET hidden size')
    parser.add_argument('--llm_ctx_size', type=int, default=2048, help='LLM hidden size')
    parser.add_argument('--max_action_steps', type=int, default=15)
    
    parser.add_argument('--llm_token_merge', type=str, default='linear', choices=['linear', 'mean'], help='How to merge image token features in LLM')
    parser.add_argument('--enc_full_graph', action='store_true', default=True)
    parser.add_argument('--graph_sprels', action='store_true', default=True)
    parser.add_argument('--use_lang2visn_attn', action='store_true', default=False)
    parser.add_argument('--use_clip_img_emb', action='store_true', default=False, help='Concatenate clip image embedding for DuET image embedding.')
    parser.add_argument('--global_cross_attn', action='store_true', default=True, help='Cross attention between vision and language in DuET.')
    parser.add_argument('--single_action_head', action='store_true', default=False, help='Single action head for DuET.')

    # Dropout Param
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--feat_dropout', type=float, default=0.3)

    # Submision configuration
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument("--submit", action='store_true', default=False)
    parser.add_argument('--no_backtrack', action='store_true', default=False)
    parser.add_argument('--detailed_output', action='store_true', default=False)

    # Training Configurations
    parser.add_argument(
        '--optim', type=str, default='adamW',
        choices=['rms', 'adam', 'adamW', 'sgd']
    )    # rms, adam
    parser.add_argument('--lr', type=float, default=0.00001, help="the learning rate")
    parser.add_argument('--decay', dest='weight_decay', type=float, default=0.)
    parser.add_argument(
        '--feedback', type=str, default='sample',
        help='How to choose next position, one of ``teacher``, ``sample`` and ``argmax``'
    )
    parser.add_argument('--epsilon', type=float, default=0.1, help='')

    # Model hyper params:
    parser.add_argument("--angle_feat_size", type=int, default=4)
    parser.add_argument('--image_feat_size', type=int, default=1408)
    parser.add_argument('--obj_feat_size', type=int, default=0)
    parser.add_argument('--views', type=int, default=36)

    # # A2C
    parser.add_argument("--gamma", default=0.0, type=float, help='reward discount factor')
    parser.add_argument(
        "--normalize", dest="normalize_loss", default="total", 
        type=str, help='batch or total'
    )
    parser.add_argument('--train_alg', 
        choices=['imitation', 'dagger'], 
        default='dagger'
    )

    args, _ = parser.parse_known_args()

    args = postprocess_args(args)

    return args


def postprocess_args(args):
    ROOTDIR = args.root_dir

    # Setup input paths
    ft_file_map = {
        'eva-clip-g': 'MP3D_eva_clip_g_can.lmdb',
    }
    args.img_ft_file = os.path.join(ROOTDIR, 'R2R', 'features', ft_file_map[args.features])

    args.connectivity_dir = os.path.join(ROOTDIR, 'R2R', 'connectivity')
    args.scan_data_dir = os.path.join(ROOTDIR, 'Matterport3D', 'v1_unzip_scans')

    args.anno_dir = os.path.join(ROOTDIR, args.dataset.upper(), 'annotations')
    args.candidate_file_dir = os.path.join(ROOTDIR, 'R2R', 'annotations', 'scanvp_candidates.json')

    # Build paths
    args.ckpt_dir = os.path.join(args.output_dir, 'ckpts')
    args.log_dir = os.path.join(args.output_dir, 'logs')
    args.pred_dir = os.path.join(args.output_dir, 'preds')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.pred_dir, exist_ok=True)

    return args

