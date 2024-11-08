"""
Here is an original implementation of Muon. 
Source: https://github.com/KellerJordan/modded-nanogpt
"""

import os

import torch
import torch.distributed as dist


def zeropower_via_svd(G, steps=None):
    U, S, V = G.svd()
    return U @ V.T


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
        B = A @ X
        X = a * X + b * B + c * A @ B
    if G.size(0) > G.size(1):
        X = X.T
    return X


zeropower_backends = dict(
    svd=zeropower_via_svd, newtonschulz5=zeropower_via_newtonschulz5
)


class Muon(torch.optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized by Newton-schulz

    Muon internally runs standard SGD-momentum, and then performs an orthogonalization post-
    processing step, in which each 2D parameter's update is replaced with the nearest orthogonal
    matrix. To efficiently orthogonalize each update, we use a Newton-Schulz iteration, which has
    the advantage that it can be stably run in bfloat16 on the GPU.

    Some warnings:
    - This optimizer assumes that all parameters passed in are 2D.
    - It should not be used for the embedding layer, the final fully connected layer, or any {0,1}-D
    parameters; those should all be optimized by a standard method (e.g., AdamW).
    - To use it with 4D convolutional filters, it works well to just flatten their last 3 dimensions.
    - We believe it is unlikely to work well for training with small batch size.
    - We believe it may not work well for finetuning pretrained models, but we haven't tested this.
    - We have not yet tried this optimizer for training scenarios larger than NanoGPT (124M).

    Arguments:
        lr: The learning rate used by the internal SGD.
        momentum: The momentum used by the internal SGD.
        nesterov: Whether to use Nesterov-style momentum in the internal SGD. (recommended)
        backend: The chosen backend for the orthogonalization step. (recommended: 'newtonschulz5')
        backend_steps: The number of iteration steps to use in the backend, if it is iterative.
    """

    def __init__(
        self,
        params,
        lr=0.02,
        momentum=0.95,
        nesterov=True,
        backend="newtonschulz5",
        backend_steps=5,
    ):
        defaults = dict(
            lr=lr,
            momentum=momentum,
            nesterov=nesterov,
            backend=backend,
            backend_steps=backend_steps,
        )
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            zeropower_backend = zeropower_backends[group["backend"]]

            # generate weight updates in distributed fashion
            total_params = sum(p.numel() for p in group["params"])
            updates_flat = torch.zeros(
                total_params, device="cuda", dtype=torch.bfloat16
            )
            curr_idx = 0
            for i, p in enumerate(group["params"]):
                # luckily this will perfectly distribute a transformer with multiple of 4 layers to 8 GPUs
                if i % int(os.environ["WORLD_SIZE"]) == int(os.environ["RANK"]):
                    g = p.grad
                    assert g is not None
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    if group["nesterov"]:
                        g = g.add(buf, alpha=momentum)
                    g = zeropower_backend(g, steps=group["backend_steps"])
                    g *= max(1, g.size(0) / g.size(1)) ** 0.5
                    updates_flat[curr_idx : curr_idx + p.numel()] = g.flatten()
                curr_idx += p.numel()

            # sync updates across devices. we are not memory-constrained so can do this simple deserialization
            dist.all_reduce(updates_flat, op=dist.ReduceOp.SUM)

            # deserialize and apply updates
            curr_idx = 0
            for p in group["params"]:
                g = (
                    updates_flat[curr_idx : curr_idx + p.numel()]
                    .view_as(p.data)
                    .type_as(p.data)
                )
                p.data.add_(g, alpha=-lr)
                curr_idx += p.numel()


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


class CombinedOptimizer(torch.optim.Optimizer):
    """
# Note that CombinedOptimizer is not a torch.optim.Optimizer, but a wrapper around multiple optimizers.
# Original Example:
    optimizer = CombinedOptimizer([
        torch.optim.AdamW(self.lm_head.parameters(), lr=learning_rate, betas=betas, weight_decay=0, fused=True),
        OrthogonalNesterov(self.transformer.h.parameters(), lr=0.1*learning_rate, momentum=0.95)
    ])
# Refactored Example:
    optimizer = CombinedOptimizer(\
        self.parameters(),
        [OrthogonalNesterov, torch.optim.AdamW],
        [{'lr': 0.1*learning_rate, 'momentum': 0.95}, 
         {'lr': learning_rate, 'betas': betas, 'weight_decay': 0, 'fused': True}
        ])
"""

    def __init__(self, params, optimizer_types, configs):
        # Separate 2D and non-2D parameters.
        # param_groups_2d_non2d: (param_groups_2d, param_groups_non2d).
        # If params is a list of tensors, then each of param_groups_2d and param_groups_non2d
        # will be a list of tensors.
        # If params is a list of dicts, then each of param_groups_2d and param_groups_non2d
        # will be a list of dicts.
        # If params is a dict, then each of param_groups_2d and param_groups_non2d will
        # be a list of dicts containing only one dict.
        (
            param_groups_2d,
            param_groups_non2d,
            total_param_2d_count,
            total_param_non2d_count,
        ) = separate_params(params)
        param_groups_2d_non2d = (param_groups_2d, param_groups_non2d)
        print(
            f"Total 2D params: {total_param_2d_count}, Total non-2D params: {total_param_non2d_count}"
        )

        assert (
            len(optimizer_types) == len(configs) == 2
        ), "You must use only two optimizers"
        assert optimizer_types[0] == Muon, "The first optimizer must be Muon"
        self.optimizers = [
            optimizer_types[i](param_groups_2d_non2d[i], **configs[i]) for i in range(2)
        ]
        self.param_groups = [pg for opt in self.optimizers for pg in opt.param_groups]
        self.base_lrs = [opt.param_groups[0]["lr"] for opt in self.optimizers]
        # Combine the state dicts of all opt in self.optimizers into a single dict
        self.state = {k: v for opt in self.optimizers for k, v in opt.state.items()}
        # Initially all states are empty. So no point to print their counts.
        # Only use the defaults of the OrthogonalNesterov optimizer
        self.defaults = self.optimizers[0].defaults

    def step(self, *args, **kwargs):
        for opt in self.optimizers:
            opt.step(*args, **kwargs)

    def zero_grad(self, **kwargs):
        for opt in self.optimizers:
            opt.zero_grad(**kwargs)

    def scale_lrs(self, lr_scale):
        for base_lr, opt in zip(self.base_lrs, self.optimizers):
            opt.param_groups[0]["lr"] = base_lr * lr_scale

    def state_dict(self):
        return [opt.state_dict() for opt in self.optimizers]
