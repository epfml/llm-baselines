"""
Here is an original implementation of ADOPT. 
Source: https://github.com/iShohei220/adopt
"""

import math
from typing import Callable, Optional, Tuple

import torch


def linear_warmup_scheduler(step, alpha_end, alpha_start=0, warmup=1):
    if step < warmup:
        a = step / float(warmup)
        return (1.0 - a) * alpha_start + a * alpha_end
    return alpha_end


def linear_hl_warmup_scheduler(step, beta_end, beta_start=0, warmup=1):
    def f(beta, eps=1e-8):
        return math.log(0.5) / math.log(beta + eps) - 1

    def f_inv(t):
        return math.pow(0.5, 1 / (t + 1))

    if step < warmup:
        a = step / float(warmup)
        return f_inv((1.0 - a) * f(beta_start) + a * f(beta_end))
    return beta_end


def exists(val):
    return val is not None


class ADOPTAdEMAMix(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.9, 0.95, 0.9999),
        alpha: float = 8.0,
        beta3_warmup=None,
        alpha_warmup=None,
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
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        self.clip_lambda = clip_lambda
        defaults = dict(
            lr=lr,
            betas=betas,
            alpha=alpha,
            beta3_warmup=beta3_warmup,
            alpha_warmup=alpha_warmup,
            eps=eps,
            weight_decay=weight_decay,
            decouple=decouple,
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
            exp_avgs_fast = []
            exp_avgs_slow = []
            exp_avg_sqs = []
            state_steps = []
            beta1, beta2, beta3_final = group["betas"]
            beta3_warmup = group["beta3_warmup"]
            alpha_final = group["alpha"]
            alpha_warmup = group["alpha_warmup"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError("ADOPT does not support sparse gradients")

                params_with_grad.append(p)
                grads.append(p.grad)

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = torch.tensor(0.0)
                    state["exp_avg_slow"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if beta1 != 0:
                        state["exp_avg_fast"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )
                    else:
                        state["exp_avg_fast"] = None
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )

                exp_avgs_fast.append(state["exp_avg_fast"])
                exp_avgs_slow.append(state["exp_avg_slow"])
                exp_avg_sqs.append(state["exp_avg_sq"])
                state_steps.append(state["step"])

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

            if beta3_warmup is not None:
                beta3 = linear_hl_warmup_scheduler(
                    state["step"],
                    beta_end=beta3_final,
                    beta_start=beta1,
                    warmup=beta3_warmup,
                )
            else:
                beta3 = beta3_final

            for i, param in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg_fast = exp_avgs_fast[i]
                exp_avg_slow = exp_avgs_slow[i]
                exp_avg_sq = exp_avg_sqs[i]
                step_t = state_steps[i]
                step = int(step_t.item())

                if step == 0:
                    exp_avg_sq.addcmul_(grad, grad)
                    step_t += 1
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

                if beta1 != 0:
                    bias_correction = 1 - beta1**step
                    exp_avg_fast.lerp_(normed_grad, 1 - beta1)
                    exp_avg_fast.div(bias_correction)
                else:
                    exp_avg_fast = normed_grad
                exp_avg_slow.lerp_(normed_grad, 1 - beta3)
                param.data.add_(exp_avg_fast + alpha * exp_avg_slow, alpha=-group["lr"])
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                step_t += 1

        return loss
