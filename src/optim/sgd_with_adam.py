# mypy: allow-untyped-defs
r"""Implementation for Stochastic Gradient Descent optimizer."""
from typing import List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim.adamw import AdamW, adamw
from torch.optim.optimizer import Optimizer  # _device_dtype_check_for_fused,
from torch.optim.optimizer import (_default_to_fused_or_foreach,
                                   _use_grad_for_differentiable)

__all__ = ["SGD", "sgd"]


class SGDWithSign(torch.optim.Optimizer):  # noqa: D101
    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        sign: bool = False,
        sign_norm: bool = False,
        normalized: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        differentiable: bool = False,
        fused: Optional[bool] = None,
    ):  # noqa: D107
        if isinstance(lr, Tensor) and lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not sign and sign_norm:
            raise ValueError(f"Enable sign to use sign_norm")
        if sign_norm and normalized:
            raise ValueError(f"These flags cancel each other out")

        if isinstance(lr, float):
            lr = torch.tensor(lr)

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            sign=sign,
            sign_norm=sign_norm,
            normalized=normalized,
            maximize=maximize,
            foreach=foreach,
            differentiable=differentiable,
            fused=fused,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

        if fused:
            self._step_supports_amp_scaling = True
            self._need_device_dtype_check_for_fused = True
            if differentiable:
                raise RuntimeError("`fused` does not support `differentiable`")
            if foreach:
                raise RuntimeError("`fused` and `foreach` cannot be `True` together.")

    def __setstate__(self, state):  # noqa: D105
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("differentiable", False)
            group.setdefault("fused", False)

    def _init_group(self, group, params, grads, momentum_buffer_list):
        has_sparse_grad = False

        for p in group["params"]:
            if p.grad is not None:
                params.append(p)
                grads.append(p.grad)
                if p.grad.is_sparse:
                    has_sparse_grad = True

                state = self.state[p]
                if "momentum_buffer" not in state:
                    momentum_buffer_list.append(None)
                else:
                    momentum_buffer_list.append(state["momentum_buffer"])

        return has_sparse_grad

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params: List[Tensor] = []
            grads: List[Tensor] = []
            momentum_buffer_list: List[Optional[Tensor]] = []

            has_sparse_grad = self._init_group(
                group, params, grads, momentum_buffer_list
            )

            sgd_advanced(
                params,
                grads,
                momentum_buffer_list,
                weight_decay=group["weight_decay"],
                momentum=group["momentum"],
                lr=group["lr"],
                dampening=group["dampening"],
                nesterov=group["nesterov"],
                sign=group["sign"],
                sign_norm=group["sign_norm"],
                normalized=group["normalized"],
                maximize=group["maximize"],
                has_sparse_grad=has_sparse_grad,
                foreach=group["foreach"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
            )

            if group["momentum"] != 0:
                # update momentum_buffers in state
                for p, momentum_buffer in zip(params, momentum_buffer_list):
                    state = self.state[p]
                    state["momentum_buffer"] = momentum_buffer

        return loss


class SGDWithAdam(torch.optim.Optimizer):  # noqa: D101
    def __init__(
        self,
        params,
        lr: Union[float, Tensor] = 1e-3,
        momentum: float = 0,
        dampening: float = 0,
        weight_decay: float = 0,
        nesterov: bool = False,
        sign: bool = False,
        sign_norm: bool = False,
        normalized: bool = False,
        adam_lr: Union[float, Tensor] = 1e-3,
        adam_betas: Tuple[float, float] = (0.9, 0.999),
        adam_eps: float = 1e-8,
        adam_amsgrad: bool = False,
        *,
        maximize: bool = False,
        foreach: Optional[bool] = None,
        fused: Optional[bool] = None,
        differentiable: bool = False,
    ):
        defaults = dict()

        if isinstance(lr, float):
            lr = torch.tensor(lr)

        super().__init__([group for group in params], defaults)
        del self.param_groups
        sgd_params = [group for group in params if self._is_sgd_group(group)]
        adam_params = [group for group in params if not self._is_sgd_group(group)]

        sgd = SGDWithSign(
            sgd_params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            maximize=maximize,
            foreach=False,
            fused=False,
            differentiable=False,
        )
        for group in sgd.param_groups:
            group.update(
                {"sign": sign, "sign_norm": sign_norm, "normalized": normalized}
            )

        adam = torch.optim.AdamW(
            adam_params,
            lr=adam_lr,
            betas=adam_betas,
            eps=adam_eps,
            weight_decay=weight_decay,
            amsgrad=adam_amsgrad,
            maximize=maximize,
            foreach=foreach,
            fused=fused,
            differentiable=differentiable,
        )
        self.param_groups = sgd.param_groups + adam.param_groups

        self._init_group_sgd_advanced = SGDWithSign._init_group.__get__(self)
        self._init_group_adam = AdamW._init_group.__get__(self)

    def _is_sgd_group(self, group):
        return group.get("is_proj_params", False) or group.get("is_sgd_params", False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if self._is_sgd_group(group):
                params: List[Tensor] = []
                grads: List[Tensor] = []
                momentum_buffer_list: List[Optional[Tensor]] = []

                has_sparse_grad = self._init_group_sgd_advanced(
                    group, params, grads, momentum_buffer_list
                )

                sgd_advanced(
                    params,
                    grads,
                    momentum_buffer_list,
                    weight_decay=group["weight_decay"],
                    momentum=group["momentum"],
                    lr=group["lr"],
                    dampening=group["dampening"],
                    nesterov=group["nesterov"],
                    sign=group["sign"],
                    sign_norm=group["sign_norm"],
                    normalized=group["normalized"],
                    maximize=group["maximize"],
                    has_sparse_grad=has_sparse_grad,
                    foreach=group["foreach"],
                    fused=group["fused"],
                    grad_scale=getattr(self, "grad_scale", None),
                    found_inf=getattr(self, "found_inf", None),
                )

                if group["momentum"] != 0:
                    # update momentum_buffers in state
                    for p, momentum_buffer in zip(params, momentum_buffer_list):
                        state = self.state[p]
                        state["momentum_buffer"] = momentum_buffer

            else:
                params_with_grad = []
                grads = []
                exp_avgs = []
                exp_avg_sqs = []
                max_exp_avg_sqs = []
                state_steps = []
                amsgrad = group["amsgrad"]
                beta1, beta2 = group["betas"]

                self._init_group_adam(
                    group,
                    params_with_grad,
                    grads,
                    amsgrad,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                )

                adamw(
                    params_with_grad,
                    grads,
                    exp_avgs,
                    exp_avg_sqs,
                    max_exp_avg_sqs,
                    state_steps,
                    amsgrad=amsgrad,
                    beta1=beta1,
                    beta2=beta2,
                    lr=group["lr"],
                    weight_decay=group["weight_decay"],
                    eps=group["eps"],
                    maximize=group["maximize"],
                    foreach=group["foreach"],
                    capturable=group["capturable"],
                    differentiable=group["differentiable"],
                    fused=group["fused"],
                    grad_scale=getattr(self, "grad_scale", None),
                    found_inf=getattr(self, "found_inf", None),
                )

        return loss


def sgd_advanced(
    params: List[Tensor],
    d_p_list: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    has_sparse_grad: bool = False,
    foreach: Optional[bool] = None,
    fused: Optional[bool] = None,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    sign: bool,
    sign_norm: bool,
    normalized: bool,
    maximize: bool,
):
    r"""Functional API that performs SGD algorithm computation.

    See :class:`~torch.optim.SGD` for details.
    """
    # Respect when the user inputs False/True for foreach or fused. We only want to change
    # the default when neither have been user-specified. Note that we default to foreach
    # and pass False to use_fused. This is not a mistake--we want to give the fused impl
    # bake-in time before making it the default, even if it is typically faster.
    if foreach is None:
        # why must we be explicit about an if statement for torch.jit.is_scripting here?
        # because JIT can't handle Optionals nor fancy conditionals when scripting
        if not torch.jit.is_scripting():
            _, foreach = _default_to_fused_or_foreach(
                params, differentiable=False, use_fused=False
            )
        else:
            foreach = False

    if foreach is None:
        foreach = False
    if fused is None:
        fused = False

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    if fused and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with fused optimizers")

    if foreach and not torch.jit.is_scripting():
        raise NotImplementedError("`foreach` option is not implemented")
        func = _multi_tensor_sgd
    elif fused and not torch.jit.is_scripting():
        raise NotImplementedError("`fused` option is not implemented")
        func = _fused_sgd
    else:
        func = _single_tensor_sgd_advanced

    func(
        params,
        d_p_list,
        momentum_buffer_list,
        weight_decay=weight_decay,
        momentum=momentum,
        lr=lr,
        dampening=dampening,
        nesterov=nesterov,
        sign=sign,
        sign_norm=sign_norm,
        normalized=normalized,
        has_sparse_grad=has_sparse_grad,
        maximize=maximize,
        grad_scale=grad_scale,
        found_inf=found_inf,
    )


def _single_tensor_sgd_advanced(
    params: List[Tensor],
    grads: List[Tensor],
    momentum_buffer_list: List[Optional[Tensor]],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    weight_decay: float,
    momentum: float,
    lr: float,
    dampening: float,
    nesterov: bool,
    sign: bool,
    sign_norm: bool,
    normalized: bool,
    maximize: bool,
    has_sparse_grad: bool,
):
    assert grad_scale is None and found_inf is None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]

        if weight_decay != 0:
            param.mul_(1 - lr * weight_decay)

        if momentum != 0:
            buf = momentum_buffer_list[i]

            if buf is None:
                buf = torch.clone(grad).detach()
                momentum_buffer_list[i] = buf
            else:
                buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf

        grads[i] = grad

    total_norm = None
    if normalized or sign_norm:
        norms: List[Tensor] = [torch.linalg.vector_norm(g, 2) for g in grads]
        total_norm = torch.linalg.vector_norm(torch.stack([norm for norm in norms]), 2)
        if normalized:
            lr = lr.div(total_norm)
        elif sign_norm:
            lr = lr.mul(total_norm)

    for i, param in enumerate(params):
        grad = grads[i]
        if sign:
            grad = grad.sign()
        param.add_(grad, alpha=-lr)


def prepare_proj_params(
    model,
    cfg,
    target_modules_list=None,
    proj_norms=False,
    proj_embeds=False,
    proj_logits=False,
):
    if target_modules_list is None:
        target_modules_list = [
            "attn",
            "mlp",
            "q_proj",
            "v_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
            "k_proj",
            "o_proj",
            "query",
            "value",
            "key",
            "intermediate.dense",
            "output.dense",
            "w1",
            "w2",
            "c_attn",
            "c_proj",
        ]

    proj_params = []

    if cfg.distributed_backend:
        model = model.module

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue

        if not any(target_key in module_name for target_key in target_modules_list):
            continue

        if module.weight.requires_grad:
            proj_params.append(module.weight)

    for name, p in model.named_parameters():
        if (
            ("ln" in name and proj_norms)
            or ("wte" in name and proj_embeds)
            or ("lm_head" in name and proj_logits)
        ):
            proj_params.append(p)

    id_proj_params = set(id(p) for p in proj_params)
    regular_params = [
        p for p in model.parameters() if id(p) not in id_proj_params and p.requires_grad
    ]

    proj_param_count = sum(p.numel() for p in proj_params)
    regular_param_count = sum(p.numel() for p in regular_params)

    print(f"Use {proj_param_count / 1e6}M projected parameters")
    print(f"Use {regular_param_count / 1e6}M regular parameters")

    return [
        {"params": regular_params, "is_proj_params": False},
        {"params": proj_params, "is_proj_params": True},
    ]
