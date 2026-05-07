import argparse
import os
import random

import numpy as np
import torch
import torch.backends

from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from utils.print_args import print_args


def get_parser():
    parser = argparse.ArgumentParser(description="EATA-NET")

    parser.add_argument(
        "--task_name",
        type=str,
        default="long_term_forecast",
        choices=["long_term_forecast"],
        help="only long-term forecasting is kept in the anonymized repository",
    )
    parser.add_argument("--is_training", type=int, default=1, help="1 for train+test, 0 for test only")
    parser.add_argument("--model_id", type=str, required=True, help="experiment identifier")
    parser.add_argument(
        "--model",
        type=str,
        default="EATA",
        choices=["EATA"],
        help="only EATA is kept in the anonymized repository",
    )

    parser.add_argument(
        "--data",
        type=str,
        required=True,
        choices=["ETTh1", "ETTh2", "ETTm1", "ETTm2", "custom", "Futures"],
        help="dataset name",
    )
    parser.add_argument("--root_path", type=str, required=True, help="dataset root path")
    parser.add_argument("--data_path", type=str, required=True, help="dataset file name")
    parser.add_argument(
        "--features",
        type=str,
        default="M",
        choices=["M", "MS"],
        help="M: multivariate->multivariate, MS: multivariate->univariate",
    )
    parser.add_argument("--target", type=str, default="OT", help="target column for custom datasets")
    parser.add_argument("--freq", type=str, default="h", help="time feature frequency")
    parser.add_argument("--checkpoints", type=str, default="./checkpoints/", help="checkpoint directory")

    parser.add_argument("--seq_len", type=int, default=96, help="input length")
    parser.add_argument("--label_len", type=int, default=48, help="label length used by the decoder stub")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction horizon")
    parser.add_argument("--inverse", action="store_true", help="inverse outputs before evaluation", default=False)

    parser.add_argument("--enc_in", type=int, required=True, help="number of input variables")
    parser.add_argument("--d_model", type=int, default=64, help="hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout")
    parser.add_argument("--k_lookback", type=int, default=64, help="lookback window used by EATA")
    parser.add_argument(
        "--method",
        type=str,
        default="Dynamic",
        choices=["Dynamic", "Simple", "None"],
        help="distribution shift tracer variant",
    )
    parser.add_argument("--hidden", type=int, default=28, help="forecast head hidden dimension")
    parser.add_argument("--bias", action="store_true", help="enable bias in the forecast head", default=False)
    parser.add_argument("--interact", action="store_true", help="enable GMIL interaction", default=False)

    parser.add_argument("--num_workers", type=int, default=4, help="dataloader workers")
    parser.add_argument("--itr", type=int, default=1, help="number of repeated runs")
    parser.add_argument("--train_epochs", type=int, default=10, help="training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="learning rate")
    parser.add_argument("--des", type=str, default="Exp", help="experiment description")
    parser.add_argument(
        "--lradj",
        type=str,
        default="type1",
        choices=["type1", "type2", "type3", "cosine"],
        help="learning rate schedule",
    )
    parser.add_argument("--use_amp", action="store_true", help="use automatic mixed precision", default=False)

    parser.add_argument("--use_gpu", type=int, default=1, choices=[0, 1], help="use GPU if available")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument(
        "--gpu_type",
        type=str,
        default="cuda",
        choices=["cuda", "mps", "cpu"],
        help="device backend",
    )

    return parser


def setup_runtime():
    torch_num_threads = os.environ.get("TSL_TORCH_NUM_THREADS")
    if torch_num_threads:
        torch.set_num_threads(max(1, int(torch_num_threads)))

    torch_interop_threads = os.environ.get("TSL_TORCH_INTEROP_THREADS")
    if torch_interop_threads:
        try:
            torch.set_num_interop_threads(max(1, int(torch_interop_threads)))
        except RuntimeError:
            pass

    fix_seed = 2021
    random.seed(fix_seed)
    np.random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(fix_seed)
        torch.cuda.manual_seed_all(fix_seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main():
    setup_runtime()
    parser = get_parser()
    args = parser.parse_args()

    if torch.cuda.is_available() and args.use_gpu and args.gpu_type == "cuda":
        args.device = torch.device(f"cuda:{args.gpu}")
        print("Using GPU")
    else:
        if args.use_gpu and args.gpu_type == "mps" and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            args.device = torch.device("mps")
            print("Using MPS")
        else:
            args.device = torch.device("cpu")
            print("Using CPU")

    print("Args in experiment:")
    print_args(args)

    Exp = Exp_Long_Term_Forecast

    if args.is_training:
        for ii in range(args.itr):
            exp = Exp(args)
            setting = (
                f"{args.model_id}_{args.model}_{args.data}_{args.features}"
                f"_sl{args.seq_len}_pl{args.pred_len}"
                f"_dm{args.d_model}_kb{args.k_lookback}"
                f"_hd{args.hidden}_{args.des}_{ii}"
            )

            print(f">>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>")
            exp.train(setting)

            print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
            exp.test(setting)
            if args.gpu_type == "mps":
                torch.backends.mps.empty_cache()
            elif args.gpu_type == "cuda" and torch.cuda.is_available():
                torch.cuda.empty_cache()
    else:
        exp = Exp(args)
        setting = (
            f"{args.model_id}_{args.model}_{args.data}_{args.features}"
            f"_sl{args.seq_len}_pl{args.pred_len}"
            f"_dm{args.d_model}_kb{args.k_lookback}"
            f"_hd{args.hidden}_{args.des}_0"
        )
        print(f">>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        exp.test(setting, test=1)
        if args.gpu_type == "mps":
            torch.backends.mps.empty_cache()
        elif args.gpu_type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
