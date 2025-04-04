from copy import deepcopy
import json
import logging
from pathlib import Path
import tempfile

import torch

from .utils import eval


logger = logging.getLogger(__name__)


class WeightAverager:
    def __init__(
        self,
        model,
        horizon=100,
        interval=1,
        save_dir=None,
        device=None,
        dtype=torch.float32,
        count=0,
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
        if save_dir is None:
            # Keep in tempdir
            self._tempdir = tempfile.TemporaryDirectory()
            self.save_dir = Path(self._tempdir.name)
        else:
            self.save_dir = Path(save_dir)
            self.save_dir.mkdir(parents=True, exist_ok=True)
        self.count = count
        # check if there are any checkpoints saved in the directory and set
        # num_saved to number of checkpoints with name <= count
        self.num_saved = len(
            [f for f in self.save_dir.iterdir() if f.is_file() and int(f.stem) <= count]
        )

    @torch.no_grad()
    def step(self, model, is_master_rank=True):
        # Update module with current state
        if self.count % self.interval == 0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                model = model.module
            for key, avg in self.module.state_dict().items():
                curr = model.state_dict()[key].to(device=self.device, dtype=avg.dtype)
                rate = 1 / ((self.count % self.horizon) // self.interval + 1)
                avg.copy_(torch.lerp(avg, curr, rate))

        self.count += 1

        if self.count % self.horizon == 0 and is_master_rank:
            torch.save(
                self.module.to().state_dict(),
                self.save_dir / f"{self.count}.pt",
            )
            self.num_saved += 1

    def get_latest_like(self, model):
        # Return model for latest completed horizon.
        if isinstance(model, torch.nn.parallel.DistributedDataParallel):
            model = model.module
        new_model = deepcopy(model)

        # Assumes that we saved at a specific iteration, will fail otherwise
        count = self.count - self.count % self.horizon
        latest_path = self.save_dir / f"{count}.pt"
        map_and_load_state_dict(new_model, torch.load(latest_path))

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
    curr_iter,
    tokens,
    epoch,
    model,
    weight_averager,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    if weight_averager.num_saved == 0:
        return

    val_reader.set_step(0)
    val_acc, val_loss = eval(
        weight_averager.get_latest_like(model).eval(),
        val_reader,
        cfg.device,
        max_num_batches=(
            val_reader.num_batches()
            if curr_iter == cfg.iterations or full_eval
            else cfg.eval_batches
        ),
        ctx=type_ctx,
        cfg=cfg,
    )

    if curr_iter == cfg.iterations or full_eval:
        logger.info("val wa (full) " + json.dumps({
            "iter": curr_iter,
            "tokens": tokens,
            "epoch": epoch,
            "val/full/wa/loss": val_loss,
            "val/full/wa/accuracy": val_acc,
        }))
    else:
        logger.info("val wa (sampled) " + json.dumps({
            "iter": curr_iter,
            "tokens": tokens,
            "epoch": epoch,
            "val/sampled/wa/loss": val_loss,
            "val/sampled/wa/accuracy": val_acc,
        }))

    print(
        f">WA Eval: Iter={curr_iter} "
        f"val_loss={val_loss:.3f} "
        f"val_acc={val_acc:3f}"
    )
