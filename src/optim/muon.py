"""
Here is an original implementation of Muon. 
Source: https://github.com/KellerJordan/modded-nanogpt
"""

import os

import torch
import torch.distributed as dist

from .schedule import cos_inf_schedule, cosine_wsd_decay_schedule, wsd_schedule


@torch.compile
def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' \sim Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - We believe this optimizer is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.

    Arguments:
        muon_params: The parameters to be optimized by Muon.
        lr: The learning rate. The updates will have spectral norm of `lr`. (0.02 is a good default)
        momentum: The momentum used by the internal SGD. (0.95 is a good default)
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        ns_steps: The number of Newton-Schulz iterations to run. (6 is probably always enough)
        adamw_params: The parameters to be optimized by AdamW. Any parameters in `muon_params` which are
        {0, 1}-D or are detected as being the embed or lm_head will be optimized by AdamW as well.
        adamw_lr: The learning rate for the internal AdamW.
        adamw_betas: The betas for the internal AdamW.
        adamw_eps: The epsilon for the internal AdamW.
        adamw_wd: The weight decay for the internal AdamW.
    """

    def __init__(
        self,
        muon_params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        ns_steps=6,
        adamw_params=None,
        adamw_lr=3e-4,
        adamw_betas=(0.95, 0.95),
        adamw_eps=1e-8,
        adamw_wd=0,
    ):

        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            adamw_lr=adamw_lr,
            adamw_lr_ratio=adamw_lr / lr,
            adamw_betas=adamw_betas,
            adamw_eps=adamw_eps,
            adamw_wd=adamw_wd,
        )

        params = list(muon_params)
        adamw_params = list(adamw_params) if adamw_params is not None else []
        params.extend(adamw_params)
        super().__init__(params, defaults)

        # Sort parameters into those for which we will use Muon, and those for which we will not
        for p in muon_params:
            # Use Muon for every parameter in muon_params which is >= 2D and doesn't look like an embedding or head layer
            if p.ndim >= 2 and p.size(0) < 10000:
                self.state[p]["use_muon"] = True
            # self.state[p]["use_muon"] = True
            else:
                self.state[p]["use_muon"] = False
        for p in adamw_params:
            # Do not use Muon for parameters in adamw_params
            self.state[p]["use_muon"] = False

        if "WORLD_SIZE" in os.environ:
            self.world_size = int(os.environ["WORLD_SIZE"])
            self.rank = int(os.environ["RANK"])
        else:
            self.world_size = 1
            self.rank = 0

    def step(self):

        for group in self.param_groups:

            ############################
            #           Muon           #
            ############################

            params = [p for p in group["params"] if self.state[p]["use_muon"]]
            lr = group["lr"]
            momentum = group["momentum"]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in params)
            updates_flat = torch.zeros(
                total_params, device="cuda", dtype=torch.bfloat16
            )
            curr_idx = 0
            for i, p in enumerate(params):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % self.world_size == self.rank:
                    g = p.grad
                    if g.ndim > 2:
                        g = g.view(g.size(0), -1)
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_via_newtonschulz5(g, steps=group["ns_steps"])
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr_idx : curr_idx + p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            if self.world_size > 1:
                dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in params:
                g = (
                    updates_flat[curr_idx : curr_idx + p.numel()]
                    .view_as(p.data)
                    .type_as(p.data)
                )
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()

            ############################
            #       AdamW backup       #
            ############################

            params = [p for p in group["params"] if not self.state[p]["use_muon"]]
            lr = (
                group["adamw_lr_ratio"] * group["lr"]
            )  # in order for lr schedule to work
            beta1, beta2 = group["adamw_betas"]
            eps = group["adamw_eps"]
            weight_decay = group["adamw_wd"]

            for p in params:
                g = p.grad
                assert g is not None
                state = self.state[p]
                if "step" not in state:
                    state["step"] = 0
                    state["moment1"] = torch.zeros_like(g)
                    state["moment2"] = torch.zeros_like(g)
                state["step"] += 1
                step = state["step"]
                buf1 = state["moment1"]
                buf2 = state["moment2"]
                buf1.lerp_(g, 1 - beta1)
                buf2.lerp_(g.square(), 1 - beta2)

                g = buf1 / (eps + buf2.sqrt())

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                scale = bias_correction1 / bias_correction2**0.5
                p.data.mul_(1 - lr * weight_decay)
                p.data.add_(g, alpha=-lr / scale)


def separate_params(param_groups):
    param_groups_2d = []
    param_groups_non2d = []
    total_param_2d_count = 0
    total_param_non2d_count = 0

    # Check if param_groups is a list of dicts or list of params
    if (
        isinstance(param_groups, list) and isinstance(param_groups[0], dict)
    ) or isinstance(param_groups, dict):
        if isinstance(param_groups, dict):
            param_groups = [param_groups]
        # param_groups is a list of dicts
        for group in param_groups:
            (
                params_2d,
                params_non2d,
                param_2d_count,
                param_non2d_count,
            ) = separate_params(group["params"])
            param_group_2d = {"params": params_2d}
            param_group_non2d = {"params": params_non2d}
            # Copy the group dict and replace the 'params' key with the separated params
            for k in group.keys():
                if k != "params":
                    param_group_2d[k] = group[k]
                    param_group_non2d[k] = group[k]

            param_groups_2d.append(param_group_2d)
            param_groups_non2d.append(param_group_non2d)
            total_param_2d_count += param_2d_count
            total_param_non2d_count += param_non2d_count

        return (
            param_groups_2d,
            param_groups_non2d,
            total_param_2d_count,
            total_param_non2d_count,
        )

    elif isinstance(param_groups, list) and isinstance(param_groups[0], torch.Tensor):
        params_2d = []
        params_non2d = []
        param_group = param_groups
        # param_group is a list of param tensors
        for param in param_group:
            if param.ndim == 2:
                params_2d.append(param)
            else:
                params_non2d.append(param)
        return params_2d, params_non2d, len(params_2d), len(params_non2d)
    else:
        breakpoint()


class CombinedScheduler:
    """
    CombinedScheduler implements a scheduler for the Muon optimizer: it leverages both Muon and AdamW learning rates, and applies the same sort of scheduler for both of them.

    Arguments:
        optimizer: Muon optimizer.
        cfg: arguments used for schedulers.
        muon_lr_key: defaults["lr"] is responsible for the Muon learning rate.
        adamw_lr_key: defaults["adamw_r"] is responsible for the AdamW learning rate.
    """

    def __init__(self, optimizer, cfg, muon_lr_key="lr", adamw_lr_key="adamw_lr"):
        self.schedulers = []
        scheduler_map = {
            "cos": torch.optim.lr_scheduler.OneCycleLR,
            "linear": torch.optim.lr_scheduler.OneCycleLR,
            "cos_inf": lambda opt, lr: torch.optim.lr_scheduler.LambdaLR(
                opt,
                cos_inf_schedule(
                    n_iterations=cfg.iterations,
                    n_warmup=cfg.warmup_steps,
                    n_inf=cfg.cos_inf_steps,
                    div_factor=1e2,
                    final_div_factor=0.1,
                ),
            ),
            "wsd": lambda opt, lr: torch.optim.lr_scheduler.LambdaLR(
                opt,
                wsd_schedule(
                    n_iterations=cfg.iterations,
                    n_warmup=cfg.warmup_steps,
                    fract_decay=cfg.wsd_fract_decay,
                    init_div_factor=1e2,
                    final_lr_factor=cfg.wsd_final_lr_scale,
                    decay_type=cfg.decay_type,
                ),
            ),
            "cos_wsd": lambda opt, lr: torch.optim.lr_scheduler.LambdaLR(
                opt,
                cosine_wsd_decay_schedule(
                    n_iterations=cfg.iterations,
                    n_warmup=cfg.warmup_steps,
                    anneal_end_factor=0.15,
                    fract_decay=cfg.wsd_fract_decay,
                    init_div_factor=1e2,
                    final_lr_factor=0.1,
                    decay_type=cfg.decay_type,
                ),
            ),
        }

        for group in optimizer.param_groups:
            lr_key = muon_lr_key if muon_lr_key in group else adamw_lr_key
            if lr_key in group:
                scheduler_cls = scheduler_map.get(cfg.scheduler, None)
                if scheduler_cls:
                    if cfg.scheduler in ["cos", "linear"]:
                        scheduler = scheduler_cls(
                            optimizer,
                            max_lr=[group.get(lr_key, getattr(cfg, lr_key.lower()))],
                            total_steps=cfg.iterations,
                            pct_start=cfg.warmup_steps / cfg.iterations,
                            anneal_strategy=cfg.scheduler,
                            cycle_momentum=False,
                            div_factor=1e2,
                            final_div_factor=1,
                        )
                    else:
                        scheduler = scheduler_cls(
                            optimizer, group.get(lr_key, getattr(cfg, lr_key.lower()))
                        )
                    self.schedulers.append(scheduler)

    def step(self):
        for scheduler in self.schedulers:
            scheduler.step()

    def state_dict(self):
        state_dict = {}
        for i, scheduler in enumerate(self.schedulers):
            state_dict[f"scheduler_{i}"] = scheduler.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        for i, scheduler in enumerate(self.schedulers):
            scheduler.load_state_dict(state_dict[f"scheduler_{i}"])
