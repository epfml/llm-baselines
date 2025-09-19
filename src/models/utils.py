import torch
<<<<<<< HEAD
from .llama import Llama, RMSNorm
from .base import GPTBase, LayerNorm

=======

from .base import GPTBase, LayerNorm
from .llama import Llama, RMSNorm
from .test import RMSNorm2, Test
>>>>>>> otherfork/main

BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    LayerNorm,
    RMSNorm,
<<<<<<< HEAD
=======
    RMSNorm2,
>>>>>>> otherfork/main
    torch.nn.Embedding,
)


def get_model(args):
<<<<<<< HEAD
    """ Return the right model """
    if args.model == 'base':
        model = GPTBase(args)
        return model
    elif args.model == 'llama2':
        model = Llama(args)
=======
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
>>>>>>> otherfork/main
        return model
    else:
        raise KeyError(f"Unknown model '{args.model}'.")
