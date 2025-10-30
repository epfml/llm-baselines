import torch

from .base import GPTBase, LayerNorm
from .llama import Llama, RMSNorm
from .mup import MuPGPTBase
from .mup_llama import MuPLlama
from .test import RMSNorm2, Test

BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    LayerNorm,
    RMSNorm,
    RMSNorm2,
    torch.nn.Embedding,
)


def get_model(args):
    """Return the right model"""
    if args.model == "base":
        model = GPTBase(args)
        if args.use_pretrained != "none":
            model.from_pretrained(args.use_pretrained)
        return model
    elif args.model == "llama":
        model = Llama(args)
        if args.use_pretrained != "none":
            raise NotImplementedError(
                f"Loading of pretrained models not yet implemented for model '{args.model}'."
            )
        return model
    elif args.model == "test":
        model = Test(args)
        if args.use_pretrained != "none":
            raise NotImplementedError(
                f"Loading of pretrained models not yet implemented for model '{args.model}'."
            )
        return model
    elif args.model == "mup_gpt":
        model = MuPGPTBase(args)
        if args.use_pretrained != "none":
            raise NotImplementedError(
                f"Loading of pretrained models not yet implemented for model '{args.model}'."
            )
        return model
    elif args.model == "mup_llama":
        model = MuPLlama(args)
        if args.use_pretrained != "none":
            raise NotImplementedError(
                f"Loading of pretrained models not yet implemented for model '{args.model}'."
            )
        return model
    else:
        raise KeyError(f"Unknown model '{args.model}'.")
