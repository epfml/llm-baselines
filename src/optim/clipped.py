"""
Here is an original implementation of Clip-Adam and Clip-Adagrad. 
Source: https://github.com/yaroslavkliukin/Clipped-AdaGrad-and-Adam
"""

import math

import torch


class AdamClip(torch.optim.Optimizer):
    """
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default True.
        clipping (str): "no", "local", "elementwise".  Default: "no"
        max_grad_norm (float): value to which we clip the gradient. Default: 1.0
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.0,
        correct_bias=True,
        clipping="no",
        max_grad_norm=1.0,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            clipping=clipping,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if group["clipping"] == "local":
                    torch.nn.utils.clip_grad_norm_(p, group["max_grad_norm"])

                if group["clipping"] == "elementwise":
                    torch.nn.utils.clip_grad_value_(p, group["max_grad_norm"])

                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]

                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                p.data.addcdiv_(tensor1=exp_avg, tensor2=denom, value=-step_size)

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss


class AdamClipDelayedEta(torch.optim.Optimizer):
    """
    Parameters:
        lr (float): learning rate. Default 1e-3.
        betas (tuple of 2 floats): Adams beta parameters (b1, b2). Default: (0.9, 0.999)
        eps (float): Adams epsilon. Default: 1e-6
        weight_decay (float): Weight decay. Default: 0.0
        correct_bias (bool): can be set to False to avoid correcting bias in Adam (e.g. like in Bert TF repository). Default False.
        clipping (str): "no", "local", "elementwise".  Default: "no"
        max_grad_norm (float): value to which we clip the gradient. Default: 1.0
        exp_avg_sq_value (float): value used to initialise the second moment in the first gradient update step. Default: 0.00001
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-6,
        weight_decay=0.0,
        correct_bias=False,
        clipping="no",
        max_grad_norm=1.0,
        exp_avg_sq_value=0.00001,
        etta=1.0,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0])
            )
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(
                "Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1])
            )
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            correct_bias=correct_bias,
            clipping=clipping,
            max_grad_norm=max_grad_norm,
            exp_avg_sq_value=exp_avg_sq_value,
            etta=etta,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if group["clipping"] == "local":
                    torch.nn.utils.clip_grad_norm_(p, group["max_grad_norm"])

                if group["clipping"] == "elementwise":
                    torch.nn.utils.clip_grad_value_(p, group["max_grad_norm"])

                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)
                    state["exp_avg_sq"] = (
                        torch.ones_like(p.data) * group["exp_avg_sq_value"]
                    )
                    # Gradient from previous step
                    state["prev_grad"] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq, prev_grad = (
                    state["exp_avg"],
                    state["exp_avg_sq"],
                    state["prev_grad"],
                )
                beta1, beta2 = group["betas"]

                state["step"] += 1

                exp_avg.mul_(beta1).add_(p.grad.data, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    prev_grad, prev_grad, value=(1.0 - beta2) * group["etta"]
                )
                state["prev_grad"] = p.grad.data
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]

                if group["correct_bias"]:
                    bias_correction1 = 1.0 - beta1 ** state["step"]
                    bias_correction2 = 1.0 - beta2 ** state["step"]
                    step_size = (
                        step_size * math.sqrt(bias_correction2) / bias_correction1
                    )

                p.data.addcdiv_(tensor1=exp_avg, tensor2=denom, value=-step_size)

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss


class AdagradClip(torch.optim.Optimizer):
    """Implements Adagrad algorithm with clipping.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        eps (float): Adagrad epsilon. Default: 1e-10
        weight_decay (float): Weight decay. Default: 0.0
        clipping (str): "no", "global", "local", "elementwise". Default: "local"
        max_grad_norm (bool): value to which we clip the gradient. Default: 1.0
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        eps=1e-10,
        weight_decay=0.0,
        clipping="local",
        max_grad_norm=1.0,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            clipping=clipping,
            max_grad_norm=max_grad_norm,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                if group["clipping"] == "local":
                    torch.nn.utils.clip_grad_norm_(p, group["max_grad_norm"])

                if group["clipping"] == "elementwise":
                    torch.nn.utils.clip_grad_value_(p, group["max_grad_norm"])

                grad = p.grad.data

                if grad.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1

                exp_avg_sq.addcmul_(grad, grad, value=1.0)
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]

                p.data.addcdiv_(tensor1=grad, tensor2=denom, value=-step_size)

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss


class AdaGradClipDelayedEta(torch.optim.Optimizer):
    """Implements Adagrad algorithm with clipping, delay and reweighing.
    Parameters:
        lr (float): learning rate. Default 1e-3.
        eps (float): Adagrad epsilon. Default: 1e-10
        weight_decay (float): Weight decay. Default: 0.0
        clipping (str): "no", "global", "local", "elementwise". Default: "local"
        max_grad_norm (bool): value to which we clip the gradient. Default: 1.0
        etta (float): reweighing parameter. Default: 1.0
        exp_avg_sq_value (float): initial value to imitate first gradient
    """

    def __init__(
        self,
        params,
        lr=1e-3,
        eps=1e-10,
        weight_decay=0.0,
        clipping="local",
        max_grad_norm=1.0,
        etta=1.0,
        exp_avg_sq_value=0.0001,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
        defaults = dict(
            lr=lr,
            eps=eps,
            weight_decay=weight_decay,
            clipping=clipping,
            max_grad_norm=max_grad_norm,
            etta=etta,
            exp_avg_sq_value=exp_avg_sq_value,
        )
        super().__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            if group["clipping"] == "global":
                torch.nn.utils.clip_grad_norm_(group["params"], group["max_grad_norm"])

            for p in group["params"]:
                if p.grad is None:
                    continue

                if group["clipping"] == "local":
                    torch.nn.utils.clip_grad_norm_(p, group["max_grad_norm"])

                if group["clipping"] == "elementwise":
                    torch.nn.utils.clip_grad_value_(p, group["max_grad_norm"])

                if p.grad.data.is_sparse:
                    raise RuntimeError(
                        "Adam does not support sparse gradients, please consider SparseAdam instead"
                    )

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg_sq"] = (
                        torch.ones_like(p.data) * group["exp_avg_sq_value"]
                    )
                    state["prev_grad"] = torch.zeros_like(p.data)

                exp_avg_sq, prev_grad = state["exp_avg_sq"], state["prev_grad"]
                state["step"] += 1

                exp_avg_sq.addcmul_(prev_grad, prev_grad, value=group["etta"])
                state["prev_grad"] = p.grad.data
                denom = exp_avg_sq.sqrt().add_(group["eps"])

                step_size = group["lr"]

                p.data.addcdiv_(tensor1=prev_grad, tensor2=denom, value=-step_size)

                if group["weight_decay"] > 0.0:
                    p.data.add_(p.data, alpha=-group["lr"] * group["weight_decay"])

        return loss
