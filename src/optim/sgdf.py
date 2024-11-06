"""
Here is an original implementation of SGDF. 
Source: https://arxiv.org/abs/2311.02818
"""

import torch


class SGDF(torch.optim.Optimizer):
    def __init__(self, params, lr=1, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if betas[0] < 0.0 or betas[1] < 0.0:
            raise ValueError("Invalid beta value: {}".format(betas))
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super(SGDF, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            eps = group["eps"]
            beta1, beta2 = group["betas"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                grad = p.grad.data
                if weight_decay != 0.0:
                    grad.add_(p.data, alpha=weight_decay)

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(grad)
                    state["exp_var"] = torch.zeros_like(grad)

                exp_avg = state["exp_avg"]
                exp_var = state["exp_var"]

                # Compute gradient 1st and 2nd
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)

                grad_residual = grad - exp_avg

                exp_var.mul_(beta2).addcmul_(
                    grad_residual, grad_residual, value=1 - beta2
                )

                state["step"] += 1

                # Bias correction
                bias_correction1 = 1 - beta1 ** state["step"]
                # bias_correction2 = 1 - beta2 ** state['step']
                bias_correction2 = (
                    (1 + beta1)
                    * (1 - beta2 ** state["step"])
                    / ((1 - beta1) * (1 - beta1 ** (2 * state["step"])))
                )

                exp_avg_corr = exp_avg / bias_correction1
                exp_var_corr = exp_var / bias_correction2

                # Wiener gain
                K = exp_var_corr / (
                    exp_var_corr + (grad - exp_avg_corr).pow(2).add_(eps)
                )

                grad_hat_residual = grad - exp_avg_corr
                grad_hat = exp_avg_corr + K * grad_hat_residual

                p.data.add_(grad_hat, alpha=-lr)

        return loss
