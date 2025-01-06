from typing import Dict

import torch

from .ademamix import linear_hl_warmup_scheduler, linear_warmup_scheduler


class Signum(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        sign_update=True,
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
        if momentum != 0:  # Signum check
            buf = state["momentum_buffer"]
            buf.mul_(momentum).add_(grad, alpha=1 - dampening)

            if nesterov:
                grad = grad.add(buf, alpha=momentum)
            else:
                grad = buf

        if sign_update:
            grad = grad.sign()

        return grad * (-lr)

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


class SignEMAMix(torch.optim.Optimizer):
    r"""Implements the SignEMAMix algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.9999))
            corresponding to beta_1, beta_2 in SignEMAMix
        alpha (float): SignEMAMix alpha coeficient mixing the slow and fast EMAs (default: 2)
        beta2_warmup (int, optional): number of warmup steps used to increase beta2 (default: None)
        alpha_warmup: (int, optional): number of warmup steps used to increase alpha (default: None)
        weight_decay (float, optional): weight decay as in AdamW (default: 0)
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.9999),
        alpha=2.0,
        beta2_warmup=None,
        alpha_warmup=None,
        weight_decay=0,
    ):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        defaults = dict(
            lr=lr,
            betas=betas,
            alpha=alpha,
            beta2_warmup=beta2_warmup,
            alpha_warmup=alpha_warmup,
            weight_decay=weight_decay,
        )
        super(SignEMAMix, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SignEMAMix, self).__setstate__(state)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            lmbda = group["weight_decay"]
            beta1, beta2_final = group["betas"]
            beta2_warmup = group["beta2_warmup"]
            alpha_final = group["alpha"]
            alpha_warmup = group["alpha_warmup"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("SignEMAMix does not support sparse gradients.")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    # Exponential moving average of gradient values
                    if beta1 != 0.0:  # save memory in case beta1 is 0.0
                        state["exp_avg_fast"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    else:
                        state["exp_avg_fast"] = None
                    state["exp_avg_slow"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg_fast, exp_avg_slow = (
                    state["exp_avg_fast"],
                    state["exp_avg_slow"],
                )

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]

                # Compute the effective alpha and beta3 in case warmup is used
                if alpha_warmup is not None:
                    alpha = linear_warmup_scheduler(
                        state["step"],
                        alpha_end=alpha_final,
                        alpha_start=0,
                        warmup=alpha_warmup,
                    )
                else:
                    alpha = alpha_final

                if beta2_warmup is not None:
                    beta2 = linear_hl_warmup_scheduler(
                        state["step"],
                        beta_end=beta2_final,
                        beta_start=beta1,
                        warmup=beta2_warmup,
                    )
                else:
                    beta2 = beta2_final

                # Decay the first and second moment running average coefficient
                if beta1 != 0.0:
                    exp_avg_fast.mul_(beta1).add_(grad, alpha=1 - beta1)
                else:
                    exp_avg_fast = grad
                exp_avg_slow.mul_(beta2).add_(grad, alpha=1 - beta2)

                update = (
                    exp_avg_fast.div(bias_correction1) + alpha * exp_avg_slow
                ).sign()

                # decay
                if lmbda != 0.0:
                    update.add_(p, alpha=lmbda)

                p.add_(-lr * update)

        return loss
