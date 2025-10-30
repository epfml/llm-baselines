import math
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import wandb


def get_batch(datareader, device="cpu"):
    x, y = datareader.sample_batch()
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y


@torch.no_grad()
def eval(
    model,
    reader,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
    moe=False,
    get_router_logits=False,
    cfg=None,
):
    assert model.training == False

    loss_list_val, acc_list, loss_list_aux_val = [], [], {}
    router_logits = []

    for idx in range(max_num_batches):
        x, y = get_batch(reader, device=device)
        with ctx:
            outputs = model(x, targets=y, get_logits=True, moe=moe)
        val_loss = outputs["loss"]

        loss_list_val.append(val_loss)
        acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        # auxiliary losses are optional
        for k, v in outputs["aux_losses"].items():
            loss_list_aux_val[k] = loss_list_aux_val.get(k, [])
            loss_list_aux_val[k].append(v)

        # router logits for MoE visualization
        if get_router_logits:
            # shape [layers, batch_size * sequence_length, num_experts]
            logits = outputs["router_logits"]
            # shape [max_batches, layers, batch_size * sequence_length, num_experts]
            router_logits.append(logits)

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828**val_loss
    val_aux_losses = {
        f"val/{k}": torch.stack(v).mean().item() for k, v in loss_list_aux_val.items()
    }

    if get_router_logits:
        # filter out the router logits that are not of the expected shape (happens for the last batch in
        # dataloader has a different batch size than the others)
        if cfg:
            intended_size = cfg.batch_size * cfg.sequence_length
        else:
            intended_size = x.shape[0] * x.shape[1]
        # shape [batches - 1, layers, batch_size * sequence_length, num_experts]
        router_logits = (
            torch.stack(
                [rl for rl in router_logits if rl.shape[1] == intended_size],
                dim=0,
            )
            .detach()
            .cpu()
        )

    return val_acc, val_loss, val_perplexity, val_aux_losses, router_logits


@torch.no_grad()
def eval_sweep_dropk(
    model,
    data_tensor,
    sequence_length,
    batch_size,
    n_heads,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
):
    assert model.training == False

    x_axis, y_axis_pp, y_axis_acc, y_axis_loss = (
        torch.linspace(0.0, 0.95, 15),
        [],
        [],
        [],
    )
    loss_list_val, acc_list = [], []

    for frac in x_axis:
        drop_k = int(sequence_length * frac * n_heads)
        for _ in range(max_num_batches):
            x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(
                    x, targets=y, alpha_th=None, drop_k=drop_k, get_logits=True
                )
            loss_list_val.append(outputs["ce_loss"])
            acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


@torch.no_grad()
def eval_sweep_alphath(
    model,
    data_tensor,
    sequence_length,
    batch_size,
    device="cpu",
    max_num_batches=24,
    ctx=nullcontext(),
):
    assert model.training == False

    alpha_ths, y_axis_pp, y_axis_acc, y_axis_loss = (
        [0, 1e-4, 1e-3, 1e-2, 1e-1, 2e-1, 3e-1, 4e-1, 5e-1],
        [],
        [],
        [],
    )
    loss_list_val, acc_list, x_axis = [], [], []

    for alpha_th in alpha_ths:
        frac_heads_pruned_list = []
        for _ in range(max_num_batches):
            x, y = get_batch(data_tensor, sequence_length, batch_size, device=device)
            with ctx:
                outputs = model(
                    x, targets=y, alpha_th=alpha_th, drop_k=None, get_logits=True
                )
            nph, nh = (
                outputs["num_head_pruned_per_layer"],
                outputs["num_heads_per_layer"],
            )
            frac_heads_pruned = np.sum(nph) / np.sum(
                nh
            )  # fractions of heads removed given alpha_th
            frac_heads_pruned_list.append(frac_heads_pruned)
            loss_list_val.append(outputs["ce_loss"])
            acc_list.append((outputs["logits"].argmax(-1) == y).float().mean())

        x_axis.append(np.mean(frac_heads_pruned_list))
        y_axis_acc.append(torch.stack(acc_list).mean().item())
        y_axis_loss.append(np.mean(loss_list_val))
        y_axis_pp.append(2.71828 ** y_axis_loss[-1])

    return x_axis, y_axis_acc, y_axis_pp, y_axis_loss


def save_checkpoint(model, opt, scheduler, itr, ckpt_dir: Path):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "itr": itr,
    }
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    torch.save(checkpoint, ckpt_dir / "main.pt")


def load_checkpoint(model, opt, scheduler, ckpt_path, device):
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["optimizer"])
    if scheduler is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    itr = ckpt["itr"]
    return itr


def save_worker_state(ckpt_dir: Path):
    # Dataloader, rng states
    worker_state = {
        "rng_torch_cpu": torch.random.get_rng_state(),
        "rng_torch_gpu": torch.cuda.get_rng_state(),
        "rng_np": np.random.get_state(),
        "rng_python": random.getstate(),
    }
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    ckpt_dir.mkdir(exist_ok=True, parents=True)
    torch.save(worker_state, ckpt_dir / f"worker_{rank}.pt")


def load_worker_state(ckpt_dir: Path):
    rank = 0 if not dist.is_initialized() else dist.get_rank()
    worker_state = torch.load(ckpt_dir / f"worker_{rank}.pt")
    torch.random.set_rng_state(worker_state["rng_torch_cpu"])
    torch.cuda.set_rng_state(worker_state["rng_torch_gpu"])
    np.random.set_state(worker_state["rng_np"])
    random.setstate(worker_state["rng_python"])


def get_parameter_norms(model, order=2):
    model_norm = 0
    for p in model.parameters():
        param_data = p.detach().data
        if order == float("inf"):
            param_norm = param_data.norm(p=order)
            model_norm = max(model_norm, param_norm.item())
        else:
            param_norm = param_data.norm(p=order)
            model_norm += param_norm.item() ** order

    if order != float("inf"):
        model_norm = model_norm ** (1.0 / order)

    return model_norm


def log_prodigy_lr(opt):
    effective_lrs = []

    for group in opt.param_groups:
        d = group["d"]
        lr = group["lr"]
        if group["use_bias_correction"]:
            k = group["k"]
            beta1, beta2 = group["betas"]
            bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
        else:
            bias_correction = 1
        effective_lr = d * lr * bias_correction
        effective_lrs.append(effective_lr)

    return effective_lrs


def visualize_routing(router_logits, extra_args):
    # router_logits: [batches, layers, batch_size * sequence_length, num_experts]
    logs = {}

    n_layers = extra_args.n_layer
    num_experts = extra_args.moe_num_experts
    num_experts_per_tok = extra_args.moe_num_experts_per_tok

    # histogram over all logits to see distribution
    logs["router/logits"] = wandb.Histogram(
        router_logits.type(torch.float32).flatten().cpu().numpy()
    )

    # distribution over experts for layer 0, layer n/2, n-1
    for layer in [0, n_layers // 2, n_layers - 1]:
        router_logits_layer = router_logits[:, layer]
        # shape [batches, batch_size * sequence_length, num_experts_per_tok]
        weights, selected_experts = torch.topk(
            router_logits_layer, num_experts_per_tok, dim=-1
        )
        # shape [batches, batch_size * sequence_length, num_experts_per_tok, num_experts]
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_experts)
        # For a given token, determine if it was routed to a given expert.
        # Shape: [batches, batch_size * sequence_length, num_experts]
        expert_mask, _ = torch.max(expert_mask, dim=-2)
        # shape [num_experts]
        tokens_per_expert = torch.mean(expert_mask, dim=(0, 1), dtype=torch.float32)
        layer_token_routing = {
            f"router/layer_{layer}_expert_{i}_selection": tokens_per_expert[i].item()
            for i in range(num_experts)
        }
        logs.update(layer_token_routing)
    return logs
