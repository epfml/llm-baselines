from contextlib import nullcontext
import torch
import torch.nn.functional as F
import wandb

from .utils import eval, get_batch


def train_sparse(model, opt, data, scheduler, iterations, acc_steps, batch_size, sequence_length, eval_freq, ckpt_path, extra_args):

    raise NotImplementedError("TODO")

    return None

