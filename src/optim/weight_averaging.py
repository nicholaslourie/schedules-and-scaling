from copy import deepcopy
import json
import logging
from pathlib import Path
import warnings

import torch

from .utils import eval


logger = logging.getLogger(__name__)


class WeightAverager:
    def __init__(
        self,
        model,
        horizon=100,
        interval=1,
        device=None,
        dtype=torch.float32,
    ):
        super().__init__()
        self.device = device  # Where to keep avg model
        self.dtype = dtype  # Precision for accumulation (>= float32)
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        self.module = deepcopy(model).to(dtype=self.dtype, device=device)

        assert horizon % interval == 0, "Interval should divide horizon"
        self.interval = interval
        self.horizon = horizon
        self.count = 0

    @torch.no_grad()
    def step(self, model, is_master_rank=True):
        # Update module with current state
        if self.count % self.interval == 0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            for key, avg in self.module.state_dict().items():
                curr = model.state_dict()[key].to(device=self.device, dtype=avg.dtype)
                rate = 1 / ((self.count % self.horizon) // self.interval + 1)
                # NOTE: When horizon divides count, rate == 1 and thus the old
                # avg is dropped and the new avg becomes equal to curr
                # (restarting the moving average for the next horizon).
                avg.copy_(torch.lerp(avg, curr, rate))

        self.count += 1

    def get_latest_like(self, model):
        if self.count % self.horizon != 0:
            raise RuntimeError(
                "Horizon is incomplete so averaged weights aren't"
                " available.",
            )

        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        new_model = deepcopy(model)

        map_and_load_state_dict(new_model, self.module.to().state_dict())

        return new_model


def map_and_load_state_dict(model, state_dict):
    for key, m_val in model.state_dict().items():
        for alias in (f'_orig_mod.{key}', f'_orig_mod.module.{key}'):  # handle compiled / nested model
            if key not in state_dict and alias in state_dict:
                key = alias
                break
        s_val = state_dict[key]
        m_val.copy_(s_val.to(device=m_val.device, dtype=m_val.dtype))


def eval_wa(
    split,
    curr_iter,
    tokens,
    epoch,
    model,
    weight_averager,
    reader,
    type_ctx,
    distributed_backend,
    cfg,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    if (
            weight_averager.count == 0 or
            weight_averager.count % weight_averager.horizon != 0
    ):
        warnings.warn(
            "Skipping weight averaging evaluation because the averaging"
            " horizon is only partially complete. To fix this, only"
            " evaluate weight averaging at multiples of wa_horizon.",
        )
        return

    reader.set_step(0)
    acc, loss = eval(
        weight_averager.get_latest_like(model).eval(),
        reader,
        cfg.device,
        max_num_batches=(
            reader.num_batches()
            if curr_iter == cfg.iterations or full_eval
            else cfg.eval_batches
        ),
        ctx=type_ctx,
        cfg=cfg,
    )

    if curr_iter == cfg.iterations or full_eval:
        logger.info(f"{split} wa (full) " + json.dumps({
            "iter": curr_iter,
            "tokens": tokens,
            "epoch": epoch,
            f"{split}/full/wa/loss": loss,
            f"{split}/full/wa/accuracy": acc,
        }))
    else:
        logger.info(f"{split} wa (sampled) " + json.dumps({
            "iter": curr_iter,
            "tokens": tokens,
            "epoch": epoch,
            f"{split}/sampled/wa/loss": loss,
            f"{split}/sampled/wa/accuracy": acc,
        }))

    print(
        f">WA Eval: Iter={curr_iter} "
        f"{split}_loss={loss:.3f} "
        f"{split}_acc={acc:3f}"
    )
