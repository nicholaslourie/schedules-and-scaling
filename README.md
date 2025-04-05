# Schedules and Scaling

This code is a fork of [`schedules-and-scaling`][schedules-repo]. That
repository accompanied the paper [*Scaling Laws and Compute-Optimal
Training Beyond Fixed Training Durations*][schedules-paper]. This fork
was used to run additional experiments based on that paper.

## Quickstart

Create a conda environment and install dependencies (we recommend Python
3.10):

```bash
conda create -n env python=3.10
conda activate env
pip install -r requirements.txt
```

Run a simple training on the SlimPajama 6B dataset:

```bash
python ./src/main.py
```

The above command trains a 213.34M parameters model with the Llama-style
architecture. We recommend to use the `--compile` flag that speeds up
training noticeably (up to 20% in our setup).

## LR Schedules and Weight Averaging

In order to use the cooldown schedule:

```bash
python ./src/main.py --compile --scheduler wsd --wsd-fract-decay 0.2
```

The argument `wsd-fract-decay` controls the fraction of the cooldown
phase, and the functional form of the cooldown is handled with the
argument `decay-type`.

If you want to use stochastic weight averaging:

```bash
python ./src/main.py --compile --scheduler wsd --wsd-fract-decay 0.2 --weight-average
```

With this, the averaging is done automatically in slots of 500 steps;
the model averages are all stored (beware of the disk space). The
frequency is handled via the arguments `--wa-interval` (average every k
steps) and `--wa-horizon` (the length of the horizon/window).

## FLOPS helpers

The [`flops.ipynb`](flops.ipynb) provides a few helpers and
functionalities for FLOPS computations of transformer configurations.

# Contact & Reference

To cite the original work this fork is based on, use:

```
@article{hagele2024scaling,
  author  = {
    Alexander H\"agele
    and Elie Bakouch
    and Atli Kosson
    and Loubna Ben Allal
    and Leandro Von Werra
    and Martin Jaggi
  },
  title   = {{Scaling Laws and Compute-Optimal Training Beyond Fixed Training Durations}},
  year    = {2024},
  journal = {Advances in Neural Information Processing Systems},
  url     = {http://arxiv.org/abs/2405.18392}
}
```

[schedules-repo]: https://github.com/epfml/schedules-and-scaling
[schedules-paper]: https://arxiv.org/abs/2405.18392
