import logging

from .llama import Llama, RMSNorm
from .base import GPTBase, LayerNorm
import torch


logger = logging.getLogger(__name__)

BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    LayerNorm,
    RMSNorm,
    torch.nn.Embedding,
)


def get_model(args):
    """Return the right model"""
    if args.model == "base":
        model = GPTBase(args)
        return model
    elif args.model == "llama":
        model = Llama(args)
        return model
    else:
        raise KeyError(f"Unknown model '{args.model}'.")
