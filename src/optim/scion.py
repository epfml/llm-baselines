"""
Here is an original implementation of Scion. 
Source: TBD
"""

import math

import torch

from .muon import zeropower_via_newtonschulz5

#######################################################
# Scion
#######################################################


class Norm(object):
    def lmo(self, g):
        raise NotImplementedError

    def init(self, w):
        raise NotImplementedError


class SpectralConv(Norm):
    def __init__(self, steps=9):
        self.steps = steps

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(
            g.shape
        )
        out_channels, in_channels, k, _ = g.shape
        g *= (out_channels / in_channels) ** 0.5 / (k**2)
        return g

    def init(self, w):
        w_fp = w.data.double()
        k = w.data.size(2)
        for kx in range(k):
            for ky in range(k):
                torch.nn.init.orthogonal_(w_fp[:, :, kx, ky])

        out_channels, in_channels, k, _ = w_fp.shape
        w_fp.mul_((out_channels / in_channels) ** 0.5 / (k**2))
        w.data = w_fp.to(dtype=w.data.dtype)
        return w


class ColNorm(Norm):
    def __init__(self, normalized=False):
        self.normalized = normalized

    def lmo(self, g):
        eps = 1e-8
        rms_values = (
            1 / math.sqrt(g.size(0)) * torch.sqrt(torch.sum(g**2, dim=-1, keepdim=True))
        )
        if self.normalized:
            rms_values *= g.size(1)
        g = g / (rms_values + eps)
        return g

    def init(self, w):
        dtype = w.data.dtype
        torch.nn.init.normal_(w.data)
        w.data /= w.norm(dim=-1, keepdim=True)
        w.data *= math.sqrt(w.size(0))
        if self.normalized:
            w.data /= w.size(1)
        w.data = w.data.to(dtype=dtype)
        return w


class RowNorm(Norm):
    def __init__(self, normalized=True):
        self.normalized = normalized

    def lmo(self, g):
        eps = 1e-8
        rms_values = torch.sqrt(torch.sum(g**2, dim=0, keepdim=True))
        if self.normalized:
            rms_values *= math.sqrt(g.size(-1))
        g = g / (rms_values + eps)
        return g

    def init(self, w):
        dtype = w.data.dtype
        torch.nn.init.normal_(w.data)
        w.data /= w.norm(dim=0, keepdim=True)
        if self.normalized:
            w.data /= math.sqrt(w.size(-1))
        w.data = w.data.to(dtype=dtype)
        return w


class Bias(Norm):
    def lmo(self, g):
        eps = 1e-8
        rms_values = torch.sqrt(torch.mean(g**2, dim=0, keepdim=True))
        g = g / (rms_values + eps)
        return g

    def init(self, g):
        return torch.nn.init.zeros_(g)


class Spectral(Norm):
    def __init__(self, max=False, steps=9):
        self.max = max
        self.steps = steps

    def lmo(self, g):
        g = zeropower_via_newtonschulz5(g.reshape(len(g), -1), steps=self.steps).view(
            g.shape
        )
        d_out, d_in = g.shape
        if self.max:
            g *= max(1, (d_out / d_in) ** 0.5)
        else:
            g *= (d_out / d_in) ** 0.5
        return g

    def init(self, w):
        w_fp = w.data.double()
        torch.nn.init.orthogonal_(w_fp)
        out_channels, in_channels = w_fp.shape
        if self.max:
            w_fp.mul_(max(1, (out_channels / in_channels) ** 0.5))
        else:
            w_fp.mul_((out_channels / in_channels) ** 0.5)
        w.data = w_fp.to(dtype=w.data.dtype)
        return w


class Sign(Norm):
    def __init__(self, zero_init=False, normalized=True):
        self.zero_init = zero_init
        self.normalized = normalized

    def lmo(self, g):
        if self.normalized:
            out_channels, in_channels = g.shape
            return (1 / in_channels) * torch.sign(g)
        else:
            return torch.sign(g)

    def init(self, w):
        if self.zero_init:
            torch.nn.init.zeros_(w)
        else:
            # Generate -1/fan_in or 1/fan_in uniformly at random
            out_channels, in_channels = w.shape
            w.data = (
                torch.randint(0, 2, w.shape, dtype=w.dtype, device=w.device) * 2 - 1
            )
            if self.normalized:
                w.data *= 1 / in_channels
        return w


class Auto(Norm):
    def lmo(self, g):
        if g.ndim == 4:
            return SpectralConv().lmo(g)
        elif g.ndim == 2:
            # print("Spectral used")
            return Spectral(max=True, steps=5).lmo(g)
        elif g.ndim == 1:
            return Bias().lmo(g)

    def init(self, w):
        if w.ndim == 4:
            return SpectralConv().init(w)
        elif w.ndim == 2:
            return Spectral(max=True, steps=5).init(w)
        elif w.ndim == 1:
            return Bias().init(w)


norm_dict = {
    "SpectralConv": SpectralConv,
    "ColNorm": ColNorm,
    "RowNorm": RowNorm,
    "Bias": Bias,
    "Spectral": Spectral,
    "Sign": Sign,
    "Auto": Auto,
}


class Scion(torch.optim.Optimizer):
    """Example usage:

    optim_groups = [{
            'params': self.transformer.wte.parameters(),
            'norm': 'ColNorm',
            'norm_kwargs': {},
            'scale': 1.0,
        }, {
            'params': self.transformer.h.parameters(),
                'norm': 'Spectral',
                'norm_kwargs': {'max': True},
                'scale': 3.0,
        }, {
            'params': self.lm_head.parameters(),
            'norm': 'Sign',
            'norm_kwargs': {},
            'scale': 10.0,
        }
    ]
    optimizer = Scion(optim_groups, lr=0.01, momentum=0.1)

    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=1.0,
        norm: str = "Auto",
        norm_kwargs: dict = None,
        scale=1.0,
        unconstrained=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if norm_kwargs is None:
            norm_kwargs = {}
        defaults = dict(
            lr=lr,
            momentum=momentum,
            scale=scale,
            unconstrained=unconstrained,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            scale = group["scale"]
            unconstrained = group["unconstrained"]
            if group["norm"] == "Auto":
                norm_backend = norm_dict[group["norm"]]()
            else:
                norm_backend = norm_dict[group["norm"]](**group["norm_kwargs"])
            for p in group["params"]:
                g = p.grad
                if g is None:
                    continue
                state = self.state[p]

                if momentum != 1:
                    if "momentum_buffer" not in state.keys():
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(1 - momentum).add_(g, alpha=momentum)
                    g = buf

                update = scale * norm_backend.lmo(g)
                if not unconstrained:
                    p.data.mul_(1 - lr)
                p.data.add_(update, alpha=-lr)

    def init(self):
        for group in self.param_groups:
            if group["norm"] == "Auto":
                norm_backend = norm_dict[group["norm"]]()
            else:
                norm_backend = norm_dict[group["norm"]](**group["norm_kwargs"])
            init_func = norm_backend.init
            scale = group["scale"]
            for p in group["params"]:
                init_func(p)
                p.data *= scale


class ScionLight(torch.optim.Optimizer):
    """
    Implementation of Scion which instead of storing both the gradient and the averaged gradient only stores the averaged gradient.

    OBS: Do not zero gradients since p.grad is directly used to store the gradient average in order to save memory.

    Example usage:

        optim_groups = [{
                'params': self.transformer.wte.parameters(),
                'norm': 'ColNorm',
                'norm_kwargs': {},
                'scale': 1.0,
            }, {
                'params': self.transformer.h.parameters(),
                    'norm': 'Spectral',
                    'norm_kwargs': {'max': True},
                    'scale': 3.0,
            }, {
                'params': self.lm_head.parameters(),
                'norm': 'Sign',
                'norm_kwargs': {},
                'scale': 10.0,
            }
        ]
        optimizer = ScionLight(optim_groups, lr=0.01, momentum=0.1)

    """

    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=1.0,
        norm: str = "Auto",
        norm_kwargs: dict = None,
        scale=1.0,
        unconstrained=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if norm_kwargs is None:
            norm_kwargs = {}
        defaults = dict(
            lr=lr,
            momentum=momentum,
            scale=scale,
            unconstrained=unconstrained,
            norm=norm,
            norm_kwargs=norm_kwargs,
        )
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            scale = group["scale"]
            unconstrained = group["unconstrained"]
            if group["norm"] == "Auto":
                norm_backend = norm_dict[group["norm"]]()
            else:
                norm_backend = norm_dict[group["norm"]](**group["norm_kwargs"])
            for p in group["params"]:
                G = p.grad
                if G is None:
                    continue

                update = scale * norm_backend.lmo(G)
                if not unconstrained:
                    p.data.mul_(1 - lr)
                p.data.add_(update, alpha=-lr)

                if momentum != 1:
                    G.mul_(1 - momentum)

    def init(self):
        for group in self.param_groups:
            if group["norm"] == "Auto":
                norm_backend = norm_dict[group["norm"]]()
            else:
                norm_backend = norm_dict[group["norm"]](**group["norm_kwargs"])
            init_func = norm_backend.init
            scale = group["scale"]
            for p in group["params"]:
                init_func(p)
                p.data *= scale


def zeroth_power_via_svd(G):
    U, S, V = G.svd()
    return U @ V.T


def scion_partitions(group_specs, model, cfg):
    model_prefix = model if cfg.distributed_backend is None else model.module

    weight_tying_enabled = (
        hasattr(model_prefix, "lm_head")
        and hasattr(model_prefix, "transformer")
        and hasattr(model_prefix.transformer, "wte")
        and model_prefix.lm_head.weight is model_prefix.transformer.wte.weight
    )

    new_groups = []
    tied_group, other_group = None, None

    if weight_tying_enabled:
        tied_vocab_size = model_prefix.lm_head.weight.shape[0]

        tied_group = {
            "params": [],
            "norm": "Sign",
            "norm_kwargs": {"normalized": False},
            "scale": cfg.scion_lmh_scale,
        }

        other_group = {
            "params": [],
            "norm": "Auto",  #'Spectral',
            "norm_kwargs": {"max": True},
            "scale": cfg.scion_tr_scale,
        }

        for g in group_specs:
            for p in g["params"]:
                if p is None or not isinstance(p, torch.Tensor):
                    continue
                if p.ndim >= 2 and p.shape[0] == tied_vocab_size:
                    tied_group["params"].append(p)
                else:
                    other_group["params"].append(p)

    else:
        emb_group = {
            "params": [],
            "norm": "ColNorm",
            "norm_kwargs": {},
            "scale": cfg.scion_emb_scale,
        }

        transformer_group = {
            "params": [],
            "norm": "Spectral",
            "norm_kwargs": {"max": True},
            "scale": cfg.scion_tr_scale,
        }

        lm_head_group = {
            "params": [],
            "norm": "Sign",
            "norm_kwargs": {},
            "scale": cfg.scion_lmh_scale,
        }

        for g in group_specs:
            for p in g["params"]:
                if p is None or not isinstance(p, torch.Tensor):
                    continue
                if (
                    hasattr(model_prefix, "transformer")
                    and hasattr(model_prefix.transformer, "wte")
                    and p is model_prefix.transformer.wte.weight
                ):
                    emb_group["params"].append(p)
                elif (
                    hasattr(model_prefix, "lm_head")
                    and p is model_prefix.lm_head.weight
                ):
                    lm_head_group["params"].append(p)
                else:
                    transformer_group["params"].append(p)

        new_groups.extend([emb_group, transformer_group, lm_head_group])

    if tied_group and len(tied_group["params"]) > 0:
        new_groups.append(tied_group)
    if other_group and len(other_group["params"]) > 0:
        new_groups.append(other_group)

    return new_groups
