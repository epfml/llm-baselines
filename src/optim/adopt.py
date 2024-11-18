"""
Here is an original implementation of ADOPT. 
Source: https://github.com/iShohei220/adopt
"""

import math

import torch


class ADOPT(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.9999),
        eps=1e-6,
        weight_decay=0.0,
        decoupled=False,
    ):
        """
        Args:
            params: iterable of parameters to optimize or dictionaries defining parameter groups.
            lr: learning rate.
            betas: coefficients used for computing running averages of gradient and gradient squared.
            eps: term added to the denominator to improve numerical stability.
            weight_decay: weight decay.
            decoupled: whether to use decoupled weight decay.
        """
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta values: {betas}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            decoupled=decoupled,
        )
        super(ADOPT, self).__init__(params, defaults)

    @torch.no_grad
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ADOPT does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1
                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                if group["weight_decay"] != 0:
                    if group["decoupled"]:
                        p.add_(p, alpha=-group["lr"] * group["decoupled"])
                    else:
                        grad = grad.add(p, alpha=group["weight_decay"])

                if state["step"] == 1:
                    exp_avg_sq.addcmul_(grad, grad)
                    continue

                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(
                    group["eps"]
                )

                if state["step"] == 2:
                    exp_avg.addcdiv_(grad, denom)
                else:
                    exp_avg.mul_(beta1).addcdiv_(grad, denom, value=1 - beta1)

                p.add_(exp_avg.div(bias_correction1), alpha=-group["lr"])

        return loss
