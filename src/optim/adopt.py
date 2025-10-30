"""
Here is an original implementation of ADOPT. 
Source: https://github.com/iShohei220/adopt
"""

import torch


def exists(val):
    return val is not None


from typing import Callable, Optional, Tuple

import torch


class ADOPT(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float] = (0.9, 0.9999),
        eps: float = 1e-6,
        clip_lambda: Optional[Callable[[int], float]] = lambda step: step**0.25,
        weight_decay: float = 0.0,
        decouple: bool = True,
    ):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.clip_lambda = clip_lambda
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, decouple=decouple
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            beta1, beta2 = group["betas"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("ADOPT does not support sparse gradients")

                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs.append(state["exp_avg"])
                exp_avg_sqs.append(state["exp_avg_sq"])

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg_sq = exp_avg_sqs[i]
                state = self.state[param]
                step = state["step"]

                if step == 0:
                    exp_avg_sq.addcmul_(grad, grad)
                    state["step"] += 1
                    continue

                if group["weight_decay"] != 0:
                    if group["decouple"]:
                        param.data.mul_(1 - group["lr"] * group["weight_decay"])
                    else:
                        grad = grad.add(param, alpha=group["weight_decay"])

                denom = torch.clamp(exp_avg_sq.sqrt(), group["eps"])
                normed_grad = grad.div(denom)

                if self.clip_lambda is not None:
                    clip = self.clip_lambda(step)
                    normed_grad.clamp_(-clip, clip)

                exp_avg.lerp_(normed_grad, 1 - beta1)
                param.data.add_(exp_avg, alpha=-group["lr"])
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                state["step"] += 1

        return loss
