import argparse
import json
import logging
from pathlib import Path
import random
import os
import schedulefree
import sys
import warnings

import numpy as np
import torch

import config
from data.utils import DataReader, get_dataset
import distributed
from models.utils import get_model
from optim.base import train
from optim.utils import cos_inf_schedule, wsd_schedule


logger = logging.getLogger(__name__)


def main(args):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s; %(levelname)s; %(name)s; %(message)s",
    )

    stderr_handler = logging.StreamHandler(stream=sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(formatter)
    root_logger.addHandler(stderr_handler)

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)
    args.world_size = distributed_backend.get_world_size()

    if args.full_eval_at is None:
        args.full_eval_at = []

    if args.weight_average and args.eval_interval % args.wa_horizon != 0:
        warnings.warn(
            "Weight averaging will not be evaluated at every evaluation"
            " interval because --eval-interval is not divisible by"
            " --wa-horizon.",
        )
    if args.weight_average and any(
            full_eval % args.wa_horizon != 0
            for full_eval in args.full_eval_at
    ):
        warnings.warn(
            "Weight averaging will not be evaluated in all full"
            " evaluations because --full-eval-at has integers which are"
            " not divisible by --wa-horizon.",
        )
    if args.weight_average and args.iterations % args.wa_horizon != 0:
        warnings.warn(
            "Weight averaging will not be evaluated at the last"
            " iteration because --iterations is not divisible by"
            " --wa-horizon.",
        )

    # NOTE args.seed is offset per worker in get_adjusted_args_for_process
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if "cuda" in args.device:
        torch.cuda.set_device(torch.device(args.device))
    # torch.use_deterministic_algorithms(True)  # CUBLAS_WORKSPACE_CONFIG=:4096:8

    exp_name = get_exp_name(args, distributed_backend)
    exp_dir = Path(args.results_base_folder) / exp_name

    if distributed_backend.is_master_process():
        exp_dir.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(
            filename=exp_dir / "log",
            mode="a",
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    print(f"Starting Experiment: {exp_name}")
    print(f"Experiment Directory: {exp_dir}")
    print(f"Config:\n{vars(args)}\n")
    if distributed_backend.is_master_process():
        logger.info(f"Experiment: {exp_name}")
        logger.info(f"Config: {json.dumps(vars(args))}")

    print(f"Loading dataset: '{args.dataset}'")
    datareaders = get_data_readers(args)

    if distributed_backend.is_master_process():
        logger.info("Num tokens: " + json.dumps({
            split: reader.num_tokens
            for split, reader in datareaders.items()
        }))

    model = get_model(args).to(args.device)
    # TODO: take care of initializing the model if args.use_pretrained != 'none'
    print(f"\nModel:\n{model}")
    if distributed_backend.is_master_process():
        logger.info(f"Model:\n{model}")

    model = distributed_backend.transform_model(model)
    group_specs = distributed_backend.get_raw_model(model).get_parameter_group_specs()
    param_name_mapping = {p_name: p for p_name, p in model.named_parameters()}
    optimized_params_cnt = 0
    for g in group_specs:
        params = []
        for p_name in g["params"]:
            translated_p_names = (
                distributed_backend.translate_model_parameter_name_for_node(p_name)
            )
            params += [param_name_mapping[p_name] for p_name in translated_p_names]
        g["params"] = params
        optimized_params_cnt += sum([p.numel() for p in g["params"]])
    params_cnt = distributed_backend.get_raw_model(model).get_num_params()
    print("number of parameters: %.2fM" % (params_cnt / 1e6,))
    print("number of optimized parameters: %.2fM" % (optimized_params_cnt / 1e6,))
    if distributed_backend.is_master_process():
        logger.info("Num parameters: " + json.dumps({
            "parameters": params_cnt,
            "optimized_parameters": optimized_params_cnt,
        }))

    if args.opt == "adamw":
        opt = torch.optim.AdamW(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.opt == "SFAdamW":
        opt = schedulefree.AdamWScheduleFree(
            group_specs,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
        )

    else:
        opt = torch.optim.SGD(
            group_specs, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay
        )
    print(f"\nOptimizer:\n{opt}")
    if distributed_backend.is_master_process():
        logger.info(f"Optimizer:\n{opt}")

    if args.scheduler != "none":
        assert args.warmup_steps < args.iterations, "Warmup steps must be < iterations."
        if args.scheduler in ["cos", "linear"]:
            # initial lr is args.lr / div_factor
            # final lr is initial_lr/final_div_factor = args.lr / div_factor / final_div_factor
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer=opt,
                max_lr=[group.get("lr", args.lr) for group in group_specs],
                total_steps=args.iterations,
                pct_start=args.warmup_steps / args.iterations,
                anneal_strategy=args.scheduler,
                cycle_momentum=False,
                div_factor=1e2,
                final_div_factor=0.1,
            )
        elif args.scheduler == "cos_inf":
            lambda_schedule = cos_inf_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                n_inf=args.cos_inf_steps,
                div_factor=1e2,
                final_div_factor=0.1,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
        elif args.scheduler == "wsd":
            lambda_schedule = wsd_schedule(
                n_iterations=args.iterations,
                n_warmup=args.warmup_steps,
                fract_decay=args.wsd_fract_decay,
                init_div_factor=1e2,
                final_lr_factor=args.wsd_final_lr_scale,  # should be 0 here
                decay_type=args.decay_type,
            )
            scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lambda_schedule)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None

    if (exp_dir / "ckpts" / "latest" / "main.pt").exists():
        if not args.auto_resume:
            raise ValueError(
                f"The experiment dir {exp_dir} already exists. "
                + "To resume training, set auto_resume=True. "
                + "Otherwise, specify a different experiment name. "
            )
        else:
            # Auto resume overwrites resume_from
            args.resume_from = str(exp_dir / "ckpts" / "latest")

    stats = train(
        model=model,
        opt=opt,
        datareaders=datareaders,
        scheduler=scheduler,
        exp_dir=exp_dir,
        distributed_backend=distributed_backend,
        cfg=args,
    )

    stats["args"] = vars(args)
    if distributed_backend.is_master_process():
        with open(exp_dir / "summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--config_format", default="base", choices=config.registered_formats()
    )

    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(
        format=args.config_format, base_parser=parser, args=rem_args, namespace=args
    )


def get_exp_name(args, distributed_backend):
    """Returns the name of the experiment, used for saving models."""
    if args.experiment_name is not None:
        return args.experiment_name

    rank = distributed_backend.rank

    exp_name = (
        f"{args.dataset}_{args.model}_nlayers{args.n_layer}"
        f"_nhead{args.n_head}_lr{args.lr}"
        f"_sched_{args.scheduler}_warmup{args.warmup_steps}"
        f"_decay_{args.decay_type}_{args.wsd_fract_decay}"
        f"_iter{args.iterations}"
        f"_bs{args.batch_size}x{args.acc_steps}_ws{args.world_size}"
    )
    # for mup
    if args.model == "mup_noam":
        exp_name = (
            f"{args.dataset}_{args.model}"
            f"_opt{args.opt}"
            f"_nlayers{args.n_layer}"
            # f"_nhead{args.n_head}"
            f"_lr{args.lr}"
            f"_sched_{args.scheduler}"
            f"_decay_{args.decay_type}"
            # f"_warmup{args.warmup_steps}"
            f"_iter{args.iterations}"
            f"_init{args.init_std}_sce{args.scale_emb}"
            f"_scd{args.scale_depth}"
            # f"_bs{args.batch_size}x{args.acc_steps}_ws{args.world_size}"
        )
    exp_name += f"_seed{args.seed - rank}"
    exp_name += f"_data_seed{args.data_seed}"

    if args.weight_average:
        exp_name += f"_WA"
    if args.opt == "SFAdamW":
        exp_name += f"_beta1_{args.beta1}"
        exp_name += f"_beta2_{args.beta2}"
    return exp_name


def get_data_readers(args, verbose=True):
    data_srcs = get_dataset(args)
    train_reader = DataReader(
        data_src=data_srcs["train"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=True,
        keep_in_ram=args.data_in_ram,
    )
    val_reader = DataReader(
        data_src=data_srcs["val"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=False,  # NOTE Identical Per Rank
        keep_in_ram=args.data_in_ram,
    )
    test_reader = DataReader(
        data_src=data_srcs["test"],
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        seed=args.data_seed,
        with_replacement=False,
        auto_shard=False,  # NOTE Identical Per Rank
        keep_in_ram=args.data_in_ram,
    )

    if verbose:
        print(f"Num training tokens: {train_reader.num_tokens}")
        print(f"Num validation tokens: {val_reader.num_tokens}")
        print(f"Num test tokens: {test_reader.num_tokens}")

    return {
        "train": train_reader,
        "val": val_reader,
        "test": test_reader,
    }


if __name__ == "__main__":
    args = get_args()
    main(args)
