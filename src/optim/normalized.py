"""
Version of different optimizers with normalized update
"""

from typing import Dict

import torch


class NormalizedSGD(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        sign_update=False,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            sign_update=sign_update,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    @torch.no_grad()
    def _init_state(self, example, state=None):
        assert isinstance(example, torch.Tensor)
        assert isinstance(state, Dict) or state is None
        if state is None:
            state = {}
        state["step"] = 0
        state["momentum_buffer"] = torch.clone(example).detach()
        return state

    @torch.no_grad()
    def _compute_update(
        self, grad, state, lr, momentum, nesterov, dampening, sign_update, **kwargs
    ):
        if momentum != 0:
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf

        if sign_update:
            update = grad.sign()
            return update / (grad.norm() + 1e-8) * (-lr)

        return grad / (grad.norm() + 1e-8) * (-lr)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                state = self.state[p]

                if group["weight_decay"] != 0:
                    p.mul_(1 - group["lr"] * group["weight_decay"])

                if len(state) == 0:
                    self._init_state(example=p, state=state)
                    if not group["momentum"]:
                        state.pop("momentum_buffer", None)

                state["step"] += 1

                update = self._compute_update(
                    grad,
                    state,
                    group["lr"],
                    group["momentum"],
                    group["nesterov"],
                    group["dampening"],
                    group["sign_update"],
                )

                p.add_(update)

        return loss
