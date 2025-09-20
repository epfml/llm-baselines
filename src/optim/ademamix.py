# Starting from the torch Adam implementation
# an implementation of AdEMAMix attempting to stick as closely as possible to the Adam implementation.
# AdEMAMix adds the additional Exponential Moving Average retain more information during long context learning.
# Originally presented in the paper: https://arxiv.org/pdf/2409.03137,
# Benchmarked for good performance with this paper: 
# And used during training of Apertus.


# TODO implement fused version, by adapting code from:
# pytorch/aten/src/ATen/native/cuda/FusedAdamWKernel.cu

# TODO look more at the scheduling side, additionally Initialization of the EMA
# is also something which could be relevant to look at.

# mypy: allow-untyped-defs
from typing import cast, Optional, Union

import torch
from torch import Tensor

from torch.optim.optimizer import (
    _capturable_doc,
    _default_to_fused_or_foreach,
    _differentiable_doc,
    #_disable_dynamo_if_unsupported,
    _foreach_doc,
    _get_capturable_supported_devices,
    _get_scalar_dtype,
    _get_value,
    _maximize_doc,
    #_params_doc,
    #_stack_if_compiling,
    #_to_scalar,
    _use_grad_for_differentiable,
    _view_as_real,
    DeviceDict,
    #DeviceDtypeDict,
    Optimizer,
    ParamsT,
)



__all__ = ["AdEMAMix", "ademamix"]


# Schedulers implemented from the jax based version of AdEMAMix, adapted for torch
def _alpha_scheduler(alpha_end, alpha_start, warmup=1):
    def schedule(step):
        is_warmup = (step < warmup).astype(torch.float32)
        a = step / float(warmup)
        return is_warmup * ((1.0-a) * alpha_start + a * alpha) + alpha * (1.0-is_warmup)
    return schedule

def _beta3_scheduler(beta_end, beta_start, warmup=1): 
    def f(beta):
        return torch.log(0.5)/torch.log(beta)-1

    def f_inv(t):
        return torch.pow(0.5,1/(t+1))
    
    def schedule(step):
        is_warmup = (step < warmup).astype(torch.float32)
        a = step / float(warmup)
        return is_warmup * f_inv((1.0-a) * f(beta_start) + a * f(beta_end)) + beta_end * (1.0-is_warmup)
    return schedule




class AdEMAMix(Optimizer):
    def __init__(
        self,
        params: ParamsT,
        lr: Union[float, Tensor] = 1e-3,
        betas: tuple[Union[float, Tensor], Union[float, Tensor],Union[float, Tensor]] = (0.9, 0.999,0.9999), # beta3 chosen from paper
        alpha: Union[float,Tensor] = 5, #TODO Verify that this is an okay value. The paper describes that the value should be set between 4-10 however the github implementation uses 2.
        eps: float = 1e-8,
        weight_decay: float = 0,
        amsgrad: bool = False,
        beta3_warmup_steps: Optional[int] = None,
        beta3_start: Optional[Union[float,Tensor]] = None,
        alpha_warmup_steps: Optional[int] = None,
        alpha_start: Optional[Union[float,Tensor]] = None, 
        *,
        foreach: Optional[bool] = None,
        maximize: bool = False,
        capturable: bool = False,
        differentiable: bool = False,
        decoupled_weight_decay: bool = True, # by default this value is true, though most things should work without it
    ):
        if isinstance(lr, Tensor):
            if foreach and not capturable:
                raise ValueError(
                    "lr as a Tensor is not supported for capturable=False and foreach=True"
                )
            if lr.numel() != 1:
                raise ValueError("Tensor lr must be 1-element")
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 2: {betas[2]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not beta3_start is None and not 0.0 <= beta3_start < 1.0:
            raise ValueError(f"Invalid start value of beta3: {beta3_start}")
        if not beta3_warmup_steps is None and not 1 < beta3_warmup_steps:
            raise ValueError(f"Invalid number of warmup steps for beta3: {beta3_warmup_steps}")
        if not alpha_warmup_steps is None and not 1 < alpha_warmup_steps:
            raise ValueError(f"Invalid number of warmup steps for alpha: {alpha_warmup_steps}")
        if not (
            (isinstance(betas[0], float) and isinstance(betas[1], float) and isinstance(betas[2], float))
            or (isinstance(betas[0], Tensor) and isinstance(betas[1], Tensor) and isinstance(betas[2], Tensor))
        ):
            raise ValueError("All three betas need to be floats or Tensors")
        if ((beta3_warmup_steps is None) != (beta3_start is None)):
            raise ValueError("Both the betas_warmup_steps and beta_start need to be initialized for scheduler")
        if ((alpha_warmup_steps is None) != (alpha_start is None)):
            raise ValueError("Both the alpha_warmup_steps and alpha_start need to be initialized for schedule")
        if isinstance(betas[0], Tensor):
            if not capturable and foreach:
                raise ValueError(
                    "betas[0] as a Tensor is not supported for capturable=False and foreach=True"
                )
            if betas[0].numel() != 1:
                raise ValueError("Tensor betas[0] must be 1-element")
        if isinstance(betas[1], Tensor):
            if not capturable and foreach:
                raise ValueError(
                    "betas[1] as a Tensor is not supported for capturable=False and foreach=True"
                )
            if betas[1].numel() != 1:
                raise ValueError("Tensor betas[1] must be 1-element")
        if isinstance(betas[2], Tensor):
            if not capturable and foreach:
                raise ValueError(
                    "betas[2] as a Tensor is not supported for capturable=False and foreach=True"
                )
            if betas[2].numel() != 1:
                raise ValueError("Tensor betas[2] must be 1-element")

        defaults = dict(
            lr=lr,
            betas=betas,
            alpha=alpha,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            maximize=maximize,
            foreach=foreach,
            capturable=capturable,
            differentiable=differentiable,
            decoupled_weight_decay=decoupled_weight_decay,
        )
        self.beta3_scheduler = None if beta3_start is None else _beta3_scheduler(betas[2], beta3_start, beta3_warmup_steps)
        self.alpha_scheduler = None if alpha_start is None else _alpha_scheduler(alpha, alpha_start, alpha_warmup_steps)
        super().__init__(params, defaults)

    def __setstate__(self, state):
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault("amsgrad", False)
            group.setdefault("maximize", False)
            group.setdefault("foreach", None)
            group.setdefault("capturable", False)
            group.setdefault("differentiable", False)
            group.setdefault("decoupled_weight_decay", False)
            for p in group["params"]:
                p_state = self.state.get(p, [])
                if len(p_state) != 0 and not torch.is_tensor(p_state["step"]):
                    step_val = float(p_state["step"])
                    p_state["step"] = (
                        torch.tensor(
                            step_val,
                            dtype=_get_scalar_dtype(is_fused=False),
                            device=p.device,
                        )
                        if group["capturable"]
                        else torch.tensor(step_val, dtype=_get_scalar_dtype())
                    )

    def _init_group(
        self,
        group,
        params_with_grad,
        grads,
        exp_avgs_fast,
        exp_avgs_slow,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
    ):
        has_complex = False
        for p in group["params"]:
            if p.grad is not None:
                has_complex |= torch.is_complex(p)
                params_with_grad.append(p)
                if p.grad.is_sparse:
                    raise RuntimeError(
                        "AdEMAMix does not support sparse gradients"
                    )
                grads.append(p.grad)

                state = self.state[p]
                # Lazy state initialization
                if len(state) == 0:
                    # note(crcrpar): [special device hosting for step]
                    # Deliberately host `step` on CPU if capturable is off.
                    # This is because kernel launches are costly on CUDA and XLA.
                    state["step"] = (
                        torch.zeros(
                            (),
                            dtype=_get_scalar_dtype(is_fused=False),
                            device=p.device,
                        )
                        if group["capturable"]
                        else torch.tensor(0.0, dtype=_get_scalar_dtype())
                    )
                    # Exponential moving average of gradient values
                    state["exp_avg_fast"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Additional exponential moving average with slower updates introduced by AdEMAMix
                    state["exp_avg_slow"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(
                        p, memory_format=torch.preserve_format
                    )
                    if group["amsgrad"]:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            p, memory_format=torch.preserve_format
                        )

                exp_avgs_fast.append(state["exp_avg_fast"])
                exp_avgs_slow.append(state["exp_avg_slow"])
                exp_avg_sqs.append(state["exp_avg_sq"])

                if group["amsgrad"]:
                    max_exp_avg_sqs.append(state["max_exp_avg_sq"])
                if group["differentiable"] and state["step"].requires_grad:
                    raise RuntimeError(
                        "`requires_grad` is not supported for `step` in differentiable mode"
                    )

                # Foreach without capturable does not support a tensor lr
                if (
                    group["foreach"]
                    and torch.is_tensor(group["lr"])
                    and not group["capturable"]
                ):
                    raise RuntimeError(
                        "lr as a Tensor is not supported for capturable=False and foreach=True"
                    )

                state_steps.append(state["step"])
        return has_complex

    @_use_grad_for_differentiable
    def step(self, closure=None):
        """Perform a single optimization step.

        Args:
            closure (Callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        self._cuda_graph_capture_health_check()

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params_with_grad: list[Tensor] = []
            grads: list[Tensor] = []
            exp_avgs_fast: list[Tensor] = []
            exp_avgs_slow: list[Tensor] = []
            exp_avg_sqs: list[Tensor] = []
            max_exp_avg_sqs: list[Tensor] = []
            state_steps: list[Tensor] = []
            beta1, beta2, beta3 = group["betas"]
            alpha = group['alpha']
            if self.beta3_scheduler:
                beta3 = self.beta3_scheduler(state_steps)
            if self.alpha_scheduler:
                alpha = self.alpha_scheduler(state_steps)
            has_complex = self._init_group(
                group,
                params_with_grad,
                grads,
                exp_avgs_fast,
                exp_avgs_slow,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
            )
            ademamix(
                params_with_grad,
                grads,
                exp_avgs_fast,
                exp_avgs_slow,
                exp_avg_sqs,
                max_exp_avg_sqs,
                state_steps,
                amsgrad=group["amsgrad"],
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                beta3=beta3,
                alpha=alpha,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                decoupled_weight_decay=group["decoupled_weight_decay"],
            )

        return loss

AdEMAMix.__doc__ = (
    r"""Implements AdEMAMix algorithm.

    .. math::
       \begin{aligned}
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{input}      : \gamma \text{ (lr)}, \beta_1, \beta_2, \beta_3, \alpha
                \text{ (betas)},\theta_0 \text{ (params)},f(\theta) \text{ (objective)}          \\
            &\hspace{13mm}      \lambda \text{ (weight decay)},  \: \textit{amsgrad},
                \:\textit{maximize},  \: \epsilon \text{ (epsilon)}                              \\
            &\textbf{initialize} :  m_0 \leftarrow 0 \text{ ( first moment)},
                v_0\leftarrow 0 \text{ (second moment)},\: v_0^{max}\leftarrow 0          \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                                 \\
            &\textbf{for} \: t=1 \: \textbf{to} \: \ldots \: \textbf{do}                         \\

            &\hspace{5mm}\textbf{if} \: \textit{maximize}:                                       \\
            &\hspace{10mm}g_t           \leftarrow   -\nabla_{\theta} f_t (\theta_{t-1})         \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}g_t           \leftarrow   \nabla_{\theta} f_t (\theta_{t-1})          \\
            &\hspace{5mm}\textbf{if} \: \lambda \neq 0                                           \\
            &\hspace{10mm} g_t \leftarrow g_t + \lambda  \theta_{t-1}                            \\
            &\hspace{5mm}m_t           \leftarrow   \beta_1 m_{t-1} + (1 - \beta_1) g_t          \\
            &\hspace{5mm}v_t           \leftarrow   \beta_2 v_{t-1} + (1-\beta_2) g^2_t          \\
            &\hspace{5mm}\widehat{m_t} \leftarrow   m_t/\big(1-\beta_1^t \big)                   \\
            &\hspace{5mm}\textbf{if} \: amsgrad                                                  \\
            &\hspace{10mm} v_t^{max} \leftarrow \mathrm{max}(v_{t-1}^{max},v_t)                  \\
            &\hspace{10mm}\widehat{v_t} \leftarrow v_t^{max}/\big(1-\beta_2^t \big)              \\
            &\hspace{5mm}\textbf{else}                                                           \\
            &\hspace{10mm}\widehat{v_t} \leftarrow   v_t/\big(1-\beta_2^t \big)                  \\
            &\hspace{5mm}\theta_t \leftarrow \theta_{t-1} - \gamma \widehat{m_t}/
                \big(\sqrt{\widehat{v_t}} + \epsilon \big)                                       \\
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
            &\bf{return} \:  \theta_t                                                     \\[-1.ex]
            &\rule{110mm}{0.4pt}                                                          \\[-1.ex]
       \end{aligned}

    For further details regarding the algorithm refer to `The AdEMAMix Optimizer: Better, Faster, Older; https://arxiv.org/abs/2409.03137  `_.
    """
    + rf"""
    Args:
        lr (float, Tensor, optional): learning rate (default: 1e-3). A tensor LR
            is not yet supported for all our implementations. Please use a float
            LR if you are not also specifying capturable=True.
        betas (Tuple[float, float, float], optional): coefficients used for computing
           the fast running averages of gradient, its square and the slow running average introduced by AdEMAMix
           (default: (0.9, 0.999,0.9999))
        alpha (float, Tensor, optional): coefficient used as a replacement for the bias 
        correction parameter used for the slow running average in AdEMAMix. 
        according to the AdEMAMix paper should range from (4,10) (default: 5)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        decoupled_weight_decay (bool, optional): if True, the optimizer functions as described in the original paper
        by handling the weight decay as done in AdamW and the algorithm will not accumulate weight
            decay in the momentum nor variance. (default: True)
        beta3_start (float, Tensor, optional): If set to a value it represents the initial value,
        for the linear scheduling of the beta3 parameter, it will linearly increase to approach beta3.  (default: None)
        beta3_warmup_steps (int,optional): number of steps before moving to the final beta3 value. (default: None)
        alpha_start (float, Tensor, optional): If set to a value it represents the initial value,
        for the linear scheduling of the alpha parameter, it will linearly increase to approach alpha.  (default: None)
        alpha_warmup_steps (int,optional): number of steps before moving to the final alpha value. (default: None)
        amsgrad (bool, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        {_foreach_doc}
        {_maximize_doc}
        {_capturable_doc}
        {_differentiable_doc}
    """
)

# 
def _single_tensor_ademamix(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs_fast: list[Tensor],
    exp_avgs_slow: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    has_complex: bool,
    beta1: Union[float, Tensor],
    beta2: Union[float, Tensor],
    beta3: Union[float, Tensor],
    alpha: Union[float, Tensor],
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    decoupled_weight_decay: bool,
):
    assert grad_scale is None and found_inf is None

    if torch.jit.is_scripting():
        # this assert is due to JIT being dumb and not realizing that the ops below
        # have overloads to handle both float and Tensor lrs, so we just assert it's
        # a float since most people using JIT are using floats
        assert isinstance(lr, float)
        assert isinstance(beta1, float)
        assert isinstance(beta2, float)
        assert isinstance(beta3, float)
        assert isinstance(alpha, float)
    else:
        lr = float(lr)
        # TODO: Support nonzero-dim Tensor betas, see #147921

    # We only shuffle around the beta when it is a Tensor, otherwise, we prefer
    # treating it as a scalar.
    # Note: ensure type declaration is under conditional check for isinstance
    # or else torchscript will get cranky about the DeviceDict type.
    if isinstance(beta1, Tensor):
        beta1_dict = {(beta1.device, beta1.dtype): beta1}
    else:
        beta1_dict = None
    if isinstance(beta3, Tensor):
        beta3_dict = {(beta3.device, beta3.dtype): beta3}
    else:
        beta3_dict = None

    for i, param in enumerate(params):
        grad = grads[i] if not maximize else -grads[i]
        exp_avg_fast = exp_avgs_fast[i]
        exp_avg_slow = exp_avgs_slow[i]
        exp_avg_sq = exp_avg_sqs[i]
        step_t = state_steps[i]

        # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
        if not torch.compiler.is_compiling() and capturable:
            capturable_supported_devices = _get_capturable_supported_devices()
            assert (
                param.device.type == step_t.device.type
                and param.device.type in capturable_supported_devices
            ), (
                f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."
            )

        # update step
        step_t += 1

        if weight_decay != 0:
            if decoupled_weight_decay:
                # Perform stepweight decay
                param.mul_(1 - lr * weight_decay)
            else:
                # Nested if is necessary to bypass jitscript rules
                if differentiable and isinstance(weight_decay, Tensor):
                    if weight_decay.requires_grad:
                        grad = grad.addcmul_(param.clone(), weight_decay)
                    else:
                        grad = grad.add(param, alpha=weight_decay)
                else:
                    grad = grad.add(param, alpha=weight_decay)

        if torch.is_complex(param):
            grad = torch.view_as_real(grad)
            exp_avg_fast = torch.view_as_real(exp_avg_fast)
            exp_avg_slow = torch.view_as_real(exp_avg_slow)
            exp_avg_sq = torch.view_as_real(exp_avg_sq)
            if amsgrad:
                max_exp_avg_sqs[i] = torch.view_as_real(max_exp_avg_sqs[i])
            param = torch.view_as_real(param)

        device = param.device

        if beta1_dict is not None:
            dtype = param.dtype  # type: ignore[union-attr]

            # cast to workaround https://github.com/pytorch/pytorch/issues/140601
            key = (device, dtype)
            if key not in beta1_dict:
                beta1_dict[key] = beta1.to(  # type: ignore[union-attr]
                    device=device, dtype=dtype, non_blocking=True
                )

            device_beta1: Union[float, Tensor] = beta1_dict[key]
        else:
            device_beta1 = beta1

        if beta3_dict is not None:
            dtype = param.dtype  # type: ignore[union-attr]

            # cast to workaround https://github.com/pytorch/pytorch/issues/140601
            key = (device, dtype)
            if key not in beta3_dict:
                beta3_dict[key] = beta3.to(  # type: ignore[union-attr]
                    device=device, dtype=dtype, non_blocking=True
                )

            device_beta3: Union[float, Tensor] = beta3_dict[key]
        else:
            device_beta3 = beta3

        # Decay the first and second moment running average coefficients
        exp_avg_fast.lerp_(grad, 1 - device_beta1)
        exp_avg_slow.lerp_(grad, 1 - device_beta3)

        # Nested if is necessary to bypass jitscript rules
        # CHANGE from origin, where beta2.requires_grad was split of from the other checks?
        # is it necessary for some compiler stuff then change back
        if differentiable and isinstance(beta2, Tensor) and beta2.requires_grad:
                # Using lerp to only use 2 operations bc addcmul's value cannot be a tensor
                # Showing equivalence of differentiable path and nondifferentiable path
                # expavg * b2 + grad^2 * (1-b2)
                #           add expavg * (1-b2) - expavg * (1-b2) = 0
                # expavg * b2 + expavg * (1-b2) - expavg * (1-b2) + grad^2 * (1-b2)
                # expavg - expavg * (1-b2) + grad^2 * (1-b2)
                # expavg + (grad^2 - expavg) * (1-b2)
                # expavg.lerp(grad^2, 1-beta2)    
            exp_avg_sq.lerp_(torch.square(grad), weight=1 - beta2)
        else:
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

        if capturable or differentiable:
            step = step_t

            # Nested if is necessary to bypass jitscript rules
            if differentiable and isinstance(beta1, Tensor):
                if beta1.requires_grad:
                    bias_correction1 = 1 - beta1 ** step.clone()
                else:
                    bias_correction1 = 1 - beta1**step
            else:
                bias_correction1 = 1 - beta1**step

            # Nested if is necessary to bypass jitscript rules
            if differentiable and isinstance(beta2, Tensor):
                if beta2.requires_grad:
                    bias_correction2 = 1 - beta2 ** step.clone()
                else:
                    bias_correction2 = 1 - beta2**step
            else:
                bias_correction2 = 1 - beta2**step

            # Bias correction is not used by beta3 as it is instead scaled by Alpha

            bias_correction2_sqrt = bias_correction2.sqrt()

            # Computes numerator for update step
            # As the slow moving average should not be modified by the bias correction, 
            # we cannot fold in the operation with the lr
            numer = torch.div(exp_avg_fast.clone(), bias_correction1)
            if differentiable and isinstance(alpha,torch.Tensor):
                numer.add_(alpha.clone()*exp_avg_slow.clone())
            else:
                numer.add_(alpha*exp_avg_slow.clone())

            
            # Computes denominator for update step
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                if differentiable:
                    max_exp_avg_sq = max_exp_avg_sqs[i].clone()
                else:
                    max_exp_avg_sq = max_exp_avg_sqs[i]

                max_exp_avg_sqs[i].copy_(torch.maximum(max_exp_avg_sq, exp_avg_sq))

                # Uses the max. for normalizing running avg. of gradient
                # Folds in (admittedly ugly) 1-elem step_size math here to avoid extra param-set-sized read+write
                # (can't fold it into addcdiv_ below because addcdiv_ requires value is a Number, not a Tensor)
                denom = (
                    max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt
                ).add_(eps)
            else: 
                denom = (
                    exp_avg_sq.sqrt() / bias_correction2_sqrt
                ).add_(eps)

            # lr 
            param.addcdiv_(numer, denom,value=-lr)
        else:
            step = _get_value(step_t)

            bias_correction1 = 1 - beta1**step
            bias_correction2 = 1 - beta2**step
            # No bias correction for exp_avg_slow

            step_size = lr / bias_correction1

            bias_correction2_sqrt = bias_correction2**0.5

            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sqs[i], exp_avg_sq, out=max_exp_avg_sqs[i])

                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sqs[i].sqrt() / bias_correction2_sqrt).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / bias_correction2_sqrt).add_(eps)
            numer = torch.add(exp_avg_fast,exp_avg_slow,alpha=alpha*bias_correction1)
            param.addcdiv_(numer, denom, value=-step_size)

        # Lastly, switch back to complex view
        if amsgrad and torch.is_complex(params[i]):
            max_exp_avg_sqs[i] = torch.view_as_complex(max_exp_avg_sqs[i])


def _multi_tensor_ademamix(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs_fast: list[Tensor],
    exp_avgs_slow: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    grad_scale: Optional[Tensor],
    found_inf: Optional[Tensor],
    *,
    amsgrad: bool,
    has_complex: bool,
    beta1: Union[float, Tensor],
    beta2: Union[float, Tensor],
    beta3: Union[float, Tensor],
    alpha: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    eps: float,
    maximize: bool,
    capturable: bool,
    differentiable: bool,
    decoupled_weight_decay: bool,
):
    if len(params) == 0:
        return

    if isinstance(lr, Tensor):
        if not capturable:
            raise RuntimeError(
                "lr as a Tensor is not supported for capturable=False and foreach=True"
            )
        if lr.numel() != 1:
            raise ValueError("Tensor lr must be 1-element")

    if isinstance(beta1, Tensor):
        if not capturable:
            raise ValueError(
                "beta1 as a Tensor is not supported for capturable=False and foreach=True"
            )
        if beta1.numel() != 1:
            raise ValueError("Tensor beta1 must be 1-element")

    if isinstance(beta2, Tensor):
        if not capturable:
            raise ValueError(
                "beta2 as a Tensor is not supported for capturable=False and foreach=True"
            )
        if beta2.numel() != 1:
            raise ValueError("Tensor beta2 must be 1-element")
    
    if isinstance(beta3, Tensor):
        if not capturable:
            raise ValueError(
                "beta3 as a Tensor is not supported for capturable=False and foreach=True"
            )
        if beta2.numel() != 1:
            raise ValueError("Tensor beta3 must be 1-element")

    # If compiling, the compiler will handle cudagraph checks, see note [torch.compile x capturable]
    if not torch.compiler.is_compiling() and capturable:
        capturable_supported_devices = _get_capturable_supported_devices(
            supports_xla=False
        )
        assert all(
            p.device.type == step.device.type
            and p.device.type in capturable_supported_devices
            for p, step in zip(params, state_steps)
        ), (
            f"If capturable=True, params and state_steps must be on supported devices: {capturable_supported_devices}."
        )

    assert grad_scale is None and found_inf is None

    assert not differentiable, "_foreach ops don't support autograd"

    lr = float(lr)
    # TODO: Support nonzero-dim Tensor betas, see #147921
    alpha = float(alpha)
    # TODO: Support Tensor alphas

    grouped_tensors = Optimizer._group_tensors_by_device_and_dtype(
        [params, grads, exp_avgs_fast, exp_avgs_slow, exp_avg_sqs, max_exp_avg_sqs, state_steps]  # type: ignore[list-item]
    )

    # We only shuffle around the beta when it is a Tensor and on CUDA, otherwise, we prefer
    # treating it as a scalar.
    beta1_dict: Optional[DeviceDict] = (  # type: ignore[attr-defined]
        {beta1.device: beta1}
        if isinstance(beta1, Tensor) and str(beta1.device) != "cpu"
        else None
    )
    beta3_dict: Optional[DeviceDict] = (  # type: ignore[attr-defined]
        {beta3.device: beta3}
        if isinstance(beta3, Tensor) and str(beta3.device) != "cpu"
        else None
    )

    for (
        device_params_,
        device_grads_,
        device_exp_avgs_fast_,
        device_exp_avgs_slow_,
        device_exp_avg_sqs_,
        device_max_exp_avg_sqs_,
        device_state_steps_,
    ), _ in grouped_tensors.values():
        device_params = cast(list[Tensor], device_params_)
        device_grads = cast(list[Tensor], device_grads_)
        device_exp_avgs_fast = cast(list[Tensor], device_exp_avgs_fast_)
        device_exp_avgs_slow = cast(list[Tensor], device_exp_avgs_slow_)
        device_exp_avg_sqs = cast(list[Tensor], device_exp_avg_sqs_)
        device_state_steps = cast(list[Tensor], device_state_steps_)

        device = device_params[0].device
        if beta1_dict is not None and device not in beta1_dict:
            beta1_dict[device] = beta1.to(device=device, non_blocking=True)  # type: ignore[union-attr, attr-defined]
        if beta3_dict is not None and device not in beta3_dict:
            beta3_dict[device] = beta3.to(device=device, non_blocking=True)  # type: ignore[union-attr, attr-defined]

        device_beta1 = beta1_dict[device] if beta1_dict else beta1
        device_beta3 = beta3_dict[device] if beta3_dict else beta3

        # Handle complex parameters
        if has_complex:
            if amsgrad:
                device_max_exp_avg_sqs = cast(list[Tensor], device_max_exp_avg_sqs_)
                _view_as_real(
                    device_params,
                    device_grads,
                    device_exp_avgs_fast,
                    device_exp_avgs_slow,
                    device_exp_avg_sqs,
                    device_max_exp_avg_sqs,
                )
            else:
                _view_as_real(
                    device_params, device_grads, device_exp_avgs_fast, device_exp_avgs_slow, device_exp_avg_sqs
                )

        if maximize:
            device_grads = torch._foreach_neg(device_grads)  # type: ignore[assignment]

        # Update steps
        # If steps are on CPU, foreach will fall back to the slow path, which is a for-loop calling t.add(1) over
        # and over. 1 will then be wrapped into a Tensor over and over again, which is slower than if we just
        # wrapped it once now. The alpha is required to assure we go to the right overload.
        if not torch.compiler.is_compiling() and device_state_steps[0].is_cpu:
            torch._foreach_add_(
                device_state_steps, torch.tensor(1.0, device="cpu"), alpha=1.0
            )
        else:
            torch._foreach_add_(device_state_steps, 1)

        if weight_decay != 0:
            if decoupled_weight_decay:
                # Perform stepweight decay
                torch._foreach_mul_(device_params, 1 - lr * weight_decay)
            else:
                # Re-use the intermediate memory (device_grads) already allocated for maximize
                if maximize:
                    torch._foreach_add_(device_grads, device_params, alpha=weight_decay)
                else:
                    device_grads = torch._foreach_add(  # type: ignore[assignment]
                        device_grads, device_params, alpha=weight_decay
                    )

        # Decay the first and second moment running average coefficient
        # Use device beta1 if beta1 is a tensor to ensure all
        # tensors are on the same device
        torch._foreach_lerp_(device_exp_avgs_fast, device_grads, 1 - device_beta1)
        
        torch._foreach_lerp_(device_exp_avgs_slow, device_grads, 1 - device_beta3)

        torch._foreach_mul_(device_exp_avg_sqs, beta2)

        # Due to the strictness of the _foreach_addcmul API, we can't have a single
        # tensor scalar as the scalar arg (only python number is supported there)
        # as a result, separate out the value mul
        # Filed https://github.com/pytorch/pytorch/issues/139795
        if isinstance(beta2, torch.Tensor):
            scaled_device_grads = torch._foreach_mul(device_grads, 1 - beta2)  # type: ignore[assignment]
            value = 1.0
        else:
            scaled_device_grads = device_grads  # type: ignore[assignment]
            value = 1 - beta2
        
        torch._foreach_addcmul_(
            device_exp_avg_sqs, scaled_device_grads, device_grads, value
        )

        # Delete the local intermediate(s) since they won't be used anymore to save on peak memory
        del device_grads
        del scaled_device_grads

        bias_correction1: Union[tuple[Tensor, ...], list[Tensor]]
        bias_correction2: Union[tuple[Tensor, ...], list[Tensor]]
        bias_correction2_sqrt: Union[tuple[Tensor, ...], list[Tensor]]

        if capturable:
            bias_correction1 = torch._foreach_pow(beta1, device_state_steps)  # type: ignore[arg-type]
            bias_correction2 = torch._foreach_pow(beta2, device_state_steps)  # type: ignore[arg-type]
            # foreach_sub doesn't allow a scalar as the first arg
            torch._foreach_sub_(bias_correction1, 1)
            torch._foreach_sub_(bias_correction2, 1)
            # we do not negate bias_correction1 as it'll need to be negated later anyway
            torch._foreach_neg_(bias_correction2)

            # foreach_div doesn't allow a scalar as the first arg
            #torch._foreach_div_(bias_correction1, lr)
            #torch._foreach_reciprocal_(bias_correction1)
            # bias correction is now lr/(bias_correction1)
            # 

            torch._foreach_sqrt_(bias_correction2)

            # Re-assign for clarity as we maintain minimal intermediates: we'll have
            # step_size = - lr / (1 - beta1 ^ t) where t = num_steps
            # bias_correction2_sqrt = sqrt(1 - beta2 ^ t)
            #step_size = bias_correction1
            bias_correction2_sqrt = bias_correction2

            if amsgrad:
                device_max_exp_avg_sqs = cast(list[Tensor], device_max_exp_avg_sqs_)
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)  # type: ignore[assignment]

                # Set intermediate to the max. for normalizing running avg. of gradient when amsgrad
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            #torch._foreach_div_(exp_avg_sq_sqrt, step_size)

            # param + (M_2 * alpha + bias_correction1 * M_1) * lr / V_1
            numer = torch._foreach_addcmul(
                device_exp_avgs_slow,
                device_exp_avgs_fast,
                bias_correction1,
                value=-1/alpha # inverted here to skip a neg operation
                )
            # at this point, exp_avg_sq_sqrt =  [sqrt(exp_avg_sq / (1 - beta2^t)) + eps]
            # and: numer = exp_avgs_slow + exp_avgs_fast / (alpha*(1-beta1^t))
            torch._foreach_addcdiv_(device_params, numer, exp_avg_sq_sqrt,value=-lr*alpha)
        else:
            bias_correction1 = [
                1 - beta1 ** _get_value(step) for step in device_state_steps
            ]
            bias_correction2 = [
                1 - beta2 ** _get_value(step) for step in device_state_steps
            ]

            #stacked_bc = _stack_if_compiling([-1*(lr/bc) for bc in bias_correction1])

            bias_correction2_sqrt = [bc**0.5 for bc in bias_correction2]  # type: ignore[arg-type]

            if amsgrad:
                device_max_exp_avg_sqs = cast(list[Tensor], device_max_exp_avg_sqs_)
                # Maintains the maximum of all 2nd moment running avg. till now
                torch._foreach_maximum_(device_max_exp_avg_sqs, device_exp_avg_sqs)

                # Use the max. for normalizing running avg. of gradient
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_max_exp_avg_sqs)
            else:
                exp_avg_sq_sqrt = torch._foreach_sqrt(device_exp_avg_sqs)

            

            torch._foreach_div_(exp_avg_sq_sqrt, bias_correction2_sqrt)
            torch._foreach_add_(exp_avg_sq_sqrt, eps)
            # Difficult to fuse as _foreach_add cannot have an alpha as a list.
            # We can fold a operation like bias_correction into addcdiv. 
            # However as we do not want to modify device_exp_avgs_slow with bias_correction this is not possible. 
            numer = torch._foreach_div(device_exp_avgs_fast, bias_correction1)
            torch._foreach_add_(numer,device_exp_avgs_slow,alpha=alpha)

            torch._foreach_addcdiv_(
                device_params,
                numer,
                exp_avg_sq_sqrt,
                value=-lr,
            )


#@_disable_dynamo_if_unsupported(single_tensor_fn=_single_tensor_ademamix)
def ademamix(
    params: list[Tensor],
    grads: list[Tensor],
    exp_avgs_fast: list[Tensor],
    exp_avgs_slow: list[Tensor],
    exp_avg_sqs: list[Tensor],
    max_exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    # kwonly args with defaults are not supported by functions compiled with torchscript issue #70627
    # setting this as kwarg for now as functional API is compiled by torch/distributed/optim
    foreach: Optional[bool] = None,
    capturable: bool = False,
    differentiable: bool = False,
    grad_scale: Optional[Tensor] = None,
    found_inf: Optional[Tensor] = None,
    has_complex: bool = False,
    decoupled_weight_decay: bool = True,
    *,
    amsgrad: bool,
    beta1: float,
    beta2: float,
    beta3: float,
    lr: Union[float, Tensor],
    weight_decay: float,
    alpha: float,
    eps: float,
    maximize: bool,
):
    r"""Functional API that performs AdEMAMix algorithm computation.

    See :class:`~torch.optim.AdEMAMix` for details.
    """
    if foreach is None:
        _, foreach = _default_to_fused_or_foreach(
            params, differentiable, use_fused=False
        )
        # Do not flip on foreach for the unsupported case where lr is a Tensor and capturable=False.
        if foreach and isinstance(lr, Tensor) and not capturable:
            foreach = False
    if foreach is None:
        foreach = False

    # this check is slow during compilation, so we skip it
    # if it's strictly needed we can add this check back in dynamo
    if not torch.compiler.is_compiling() and not all(
        isinstance(t, torch.Tensor) for t in state_steps
    ):
        raise RuntimeError(
            "API has changed, `state_steps` argument must contain a list of singleton tensors"
        )

    if foreach and torch.jit.is_scripting():
        raise RuntimeError("torch.jit.script not supported with foreach optimizers")
    

    elif foreach and not torch.jit.is_scripting():
        func = _multi_tensor_ademamix
    else:
        func = _single_tensor_ademamix

    func(
        params,
        grads,
        exp_avgs_fast,
        exp_avgs_slow,
        exp_avg_sqs,
        max_exp_avg_sqs,
        state_steps,
        amsgrad=amsgrad,
        has_complex=has_complex,
        beta1=beta1,
        beta2=beta2,
        beta3=beta3,
        alpha=alpha,
        lr=lr,
        weight_decay=weight_decay,
        eps=eps,
        maximize=maximize,
        capturable=capturable,
        differentiable=differentiable,
        grad_scale=grad_scale,
        found_inf=found_inf,
        decoupled_weight_decay=decoupled_weight_decay,
    )
