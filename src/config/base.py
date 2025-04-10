import logging

import distributed


logger = logging.getLogger(__name__)


def parse_args(base_parser, args, namespace):
    parser = base_parser
    # General training params
    parser.add_argument("--experiment-name", default=None, type=str)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--data-seed", default=1337, type=int)
    parser.add_argument("--eval-interval", default=200, type=int)
    parser.add_argument("--full-eval-at", nargs="+", type=int)
    parser.add_argument("--eval-batches", default=32, type=int)
    parser.add_argument("--device", default="cuda:0", type=str)
    parser.add_argument(
        "--distributed-backend",
        default=None,
        type=str,
        required=False,
        choices=distributed.registered_backends(),
    )
    parser.add_argument("--log-interval", default=50, type=int)

    # Checkpointing
    parser.add_argument("--results-base-folder", default="./exps", type=str)

    # Schedule
    parser.add_argument(
        "--scheduler",
        default="cos",
        choices=["linear", "cos", "wsd", "none", "cos_inf"],
    )
    parser.add_argument("--cos-inf-steps", default=0, type=int)
    # parser.add_argument("--cos-final-lr", default=1e-6, type=float)
    parser.add_argument("--iterations", default=15000, type=int)
    parser.add_argument("--warmup-steps", default=300, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    # wsd
    parser.add_argument("--wsd-final-lr-scale", default=0.0, type=float)
    parser.add_argument("--wsd-fract-decay", default=0.1, type=float)
    # parser.add_argument("--wsd-exponential-decay", action="store_true")
    parser.add_argument(
        "--decay-type",
        default="linear",
        choices=["linear", "cosine", "exp", "miror_cosine", "square", "sqrt"],
    )
    # Optimization
    parser.add_argument("--opt", default="adamw", choices=["adamw", "sgd", "SFAdamW"])
    parser.add_argument("--batch-size", default=50, type=int)
    parser.add_argument("--acc-steps", default=4, type=int)
    parser.add_argument("--weight-decay", default=1e-1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument(
        "--grad-clip", default=1.0, type=float
    )  # default value is 1.0 in NanoGPT

    # Weight Averaging
    parser.add_argument("--weight-average", action="store_true")
    parser.add_argument(
        "--wa-interval",
        default=5,
        type=int,
        help="How often to take the average (every k steps). Must divide wa-horizon.",
    )
    parser.add_argument(
        "--wa-horizon",
        default=500,
        type=int,
        help="How many consecutive steps to use for weight averages.",
    )
    parser.add_argument(
        "--wa-dtype",
        default="float32",
        type=str,
        choices=["float32", "float64"],
    )

    # Dataset params
    parser.add_argument("--datasets-dir", type=str, default="./datasets/")
    parser.add_argument(
        "--dataset",
        default="slimpajama",
        choices=[
            "slimpajama",
        ],
    )
    parser.add_argument(
        "--tokenizer", default="gpt2", type=str, choices=["gpt2", "mistral"]
    )
    parser.add_argument("--vocab-size", default=50304, type=int)
    parser.add_argument(
        "--data-in-ram", action="store_true"
    )  # force the data to RAM, mostly useless

    # Model params
    parser.add_argument(
        "--model",
        default="llama",
        choices=[
            "base",
            "llama",
        ],
    )
    parser.add_argument("--parallel-block", action="store_true")
    parser.add_argument("--init-std", default=0.02, type=float)
    parser.add_argument("--dropout", default=0.0, type=float)
    parser.add_argument("--n-head", default=12, type=int)
    parser.add_argument("--n-layer", default=24, type=int)  # depths in att + ff blocks
    parser.add_argument("--sequence-length", default=512, type=int)
    parser.add_argument(
        "--n-embd", default=768, type=int  # embedding size / hidden size ...
    )
    parser.add_argument(
        "--multiple-of",  # make SwiGLU hidden layer size multiple of large power of 2
        default=256,
        type=int,
    )
    parser.add_argument("--rmsnorm-eps", default=1e-5, type=float)
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        type=str,
        choices=["float32", "float16", "bfloat16"],
    )
    parser.add_argument("--bias", default=False, type=bool)
    parser.add_argument("--compile", action="store_true")

    return parser.parse_args(args, namespace)
