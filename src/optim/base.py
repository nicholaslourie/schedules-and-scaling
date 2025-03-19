from contextlib import nullcontext
import copy
import json
import logging
from pathlib import Path
import time
import yaml

import torch

from optim.weight_averaging import (
    WeightAverager,
    eval_wa,
)
from .utils import (
    eval,
    get_batch,
)


logger = logging.getLogger(__name__)


def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
):
    not_compiled_model = model
    if cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model)

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    curr_iter = 0

    if cfg.weight_average:
        # This does generally not support resuming training, but will work if
        # cfg.wa_interval perfectly divides the iteration number of the chkpt.
        # Otherwise, the first avg will not be correctly computed, with a bias
        # towards the first sample and missing values for earlier iterations.
        weight_averager = WeightAverager(
            not_compiled_model,
            horizon=cfg.wa_horizon,
            interval=cfg.wa_interval,
            dtype={
                "float32": torch.float32,
                "float64": torch.float64,
            }[cfg.wa_dtype],
            count=curr_iter,
        )

    substep = curr_iter * cfg.acc_steps
    train_reader = datareaders["train"]
    val_reader = datareaders["val"]
    test_reader = datareaders["test"]
    train_reader.set_step(substep)
    stats = {"train_loss": [], "val_loss": [], "val_acc": []}
    model.train()

    while curr_iter <= cfg.iterations:
        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens
        if (
            curr_iter % cfg.eval_interval == 0
            or curr_iter == cfg.iterations
            or (curr_iter in cfg.full_eval_at)
        ):
            for split, reader in [("val", val_reader), ("test", test_reader)]:
                eval_and_log(
                    split,
                    curr_iter,
                    tokens,
                    epoch,
                    model,
                    reader,
                    type_ctx,
                    distributed_backend,
                    cfg,
                    opt,
                    full_eval=(curr_iter in cfg.full_eval_at),
                )

                if curr_iter > cfg.wa_interval and cfg.weight_average:
                    eval_wa(
                        split,
                        curr_iter,
                        tokens,
                        epoch,
                        not_compiled_model,
                        weight_averager,
                        reader,
                        type_ctx,
                        distributed_backend,
                        cfg,
                        full_eval=(curr_iter in cfg.full_eval_at),
                    )

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            break

        # Train model
        t_start = time.perf_counter_ns()
        for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
            x, y = get_batch(train_reader, device=cfg.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
                ):
                    outputs = model(x, targets=y)

            loss = outputs["loss"] / cfg.acc_steps
            loss.backward()
            substep += 1

        if cfg.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if cfg.opt == "SFAdamW":
            opt.train()
        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        if cfg.weight_average:
            weight_averager.step(not_compiled_model, distributed_backend.is_master_process())
        dt = (time.perf_counter_ns() - t_start) / 1e9

        curr_iter += 1

        if (
            cfg.log_interval
            and curr_iter % cfg.log_interval == 0
            and distributed_backend.is_master_process()  # Only log on master rank
        ):
            train_loss = loss.detach().cpu().item() * cfg.acc_steps

            current_lrs = [param_group["lr"] for param_group in opt.param_groups]

            logger.info("train (sampled) " + json.dumps({
                "iter": curr_iter,
                "lr": current_lrs[0],
                "iter_dt": dt,
                "train/sampled/raw/loss": train_loss,
            }))

            print(
                f"Train: Iter={curr_iter} ({epoch:0.3f} epochs) "
                f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
                f"lr={current_lrs[0]:.2e}"
            )

    return stats


def eval_and_log(
    split,
    curr_iter,
    tokens,
    epoch,
    model,
    reader,
    type_ctx,
    distributed_backend,
    cfg,
    opt,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    model.eval()
    if cfg.opt == "SFAdamW":
        opt.eval()

    if curr_iter == cfg.iterations or full_eval:
        max_num_batches = reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # Make sure we start from the beginning (repeat the same batches).
    reader.set_step(0)
    acc, loss = eval(
        model,
        reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
        cfg=cfg,
    )

    if curr_iter == cfg.iterations or full_eval:
        logger.info(f"{split} (full) " + json.dumps({
            "iter": curr_iter,
            "tokens": tokens,
            "epoch": epoch,
            f"{split}/full/raw/loss": loss,
            f"{split}/full/raw/accuracy": acc,
        }))
    else:
        logger.info(f"{split} (sampled) " + json.dumps({
            "iter": curr_iter,
            "tokens": tokens,
            "epoch": epoch,
            f"{split}/sampled/raw/loss": loss,
            f"{split}/sampled/raw/accuracy": acc,
        }))

    print(
        f">Eval: Iter={curr_iter} ({epoch:0.3f} epochs) "
        f"{split}_loss={loss:.3f} "
        f"{split}_acc={acc:3f}"
    )

    model.train()
