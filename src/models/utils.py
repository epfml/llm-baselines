import torch
from .llama import Llama, RMSNorm
from .base import GPTBase, LayerNorm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

BLACKLIST_WEIGHT_MODULES = (
    torch.nn.LayerNorm,
    LayerNorm,
    RMSNorm,
    torch.nn.Embedding,
)


def get_model(args):
    """ Return the right model """
    if args.model == 'base':
        model = GPTBase(args)
        return model
    elif args.model == 'llama2':
        model = Llama(args)
        return model
    else:
        raise KeyError(f"Unknown model '{args.model}'.")

def get_model_and_tokenizer(args):
    """ Return the right model and tokenizer """
    if args.model == 'gpt2-pretrained':
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        return model, tokenizer
    else:
        raise KeyError(f"Unknown model '{args.model}'.")