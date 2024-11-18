"""
Here is an original implementation of ADOPT. 
Source: https://github.com/iShohei220/adopt
"""

import torch


class ADOPT(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.0,
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
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] <= 1.0:
            raise ValueError(f"Invalid beta values: {betas}")
        if eps <= 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
        )
        super(ADOPT, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("ADOPT does not support sparse gradients")

                if group["weight_decay"] != 0:
                    grad = grad.add(p, alpha=group["weight_decay"])

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

                state["step"] += 1

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]
                eps = group["eps"]

                bias_correction1 = 1 - beta1 ** state["step"]
                bias_correction2 = 1 - beta2 ** state["step"]

                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                exp_avg_sq = exp_avg_sq.div(bias_correction2).sqrt()
                denom = torch.maximum(exp_avg_sq, torch.tensor(eps, device=grad.device))

                exp_avg.mul_(beta1).add_(grad.div(denom), alpha=1 - beta1)

                step_size = group["lr"] * (exp_avg.div(bias_correction1))
                p.add_(-step_size)

        return loss
