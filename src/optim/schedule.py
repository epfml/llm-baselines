import math

import numpy as np


def cos_inf_schedule(n_iterations, n_warmup, div_factor, final_div_factor, n_inf):
    """Cosine annealing with warmup and _constant_ final_lr after cycle ended.
    Args:
        n_iterations: total number of iterations
        n_warmup: number of warmup iterations
        div_factor: initial division factor for warmup
        final_div_factor: final division factor for final lr
        n_inf: number of iterations for the final lr (constant lr after cycle ended)
    Returns:
        schedule: a function that takes the current iteration and
        returns the multiplicative factor for the learning rate
    """
    max_lr = 1.0
    base_lr = max_lr / div_factor
    final_lr = base_lr / final_div_factor

    n_anneal_steps = n_iterations - n_inf

    def schedule(step):
        if step < n_warmup:
            return (step / n_warmup) + (1 - step / n_warmup) / div_factor
        elif step < n_anneal_steps:
            t = (step - n_warmup) / (n_anneal_steps - n_warmup)
            lr = final_lr + 0.5 * (max_lr - final_lr) * (1 + np.cos(np.pi * t))
            return lr
        else:
            return final_lr

    return schedule


def wsd_schedule(
    n_iterations,
    final_lr_factor=0.0,
    n_warmup=1000,
    init_div_factor=100,
    fract_decay=0.1,
    decay_type="linear",
):
    """Warmup, hold, and decay schedule.
    Args:
        n_iterations: total number of iterations
        final_lr_factor: factor by which to reduce max_lr at the end
        warmup_fract: fraction of iterations used for warmup
        init_div_factor: initial division factor for warmup
        fract_decay: fraction of iterations used for decay
    Returns:
        schedule: a function that takes the current iteration and
        returns the multiplicative factor for the learning rate
    """
    n_anneal_steps = int(fract_decay * n_iterations)
    n_hold = n_iterations - n_anneal_steps

    def schedule(step):
        if step < n_warmup:
            return (step / n_warmup) + (1 - step / n_warmup) / init_div_factor
        elif step < n_hold:
            return 1.0
        elif step < n_iterations:
            if decay_type == "linear":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
            elif decay_type == "exp":
                return final_lr_factor ** ((step - n_hold) / n_anneal_steps)
            elif decay_type == "cosine":
                return (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
            elif decay_type == "miror_cosine":
                cosine_value = (
                    final_lr_factor
                    + (1 - final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_hold) / n_anneal_steps))
                    * 0.5
                )
                linear_value = final_lr_factor + (1 - final_lr_factor) * (
                    1 - (step - n_hold) / n_anneal_steps
                )
                return linear_value * 2 - cosine_value
            elif decay_type == "square":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - ((step - n_hold) / n_anneal_steps) ** 2
                )

            elif decay_type == "sqrt":
                return final_lr_factor + (1 - final_lr_factor) * (
                    1 - math.sqrt((step - n_hold) / n_anneal_steps)
                )

            else:
                raise ValueError(
                    f"decay type {decay_type} is not in ['cosine','miror_cosine','linear','exp']"
                )

        else:
            return final_lr_factor

    return schedule


def cosine_wsd_decay_schedule(
    n_iterations,
    n_warmup=1000,
    anneal_end_factor=0.15,
    final_lr_factor=0.0,
    init_div_factor=1e-2,
    fract_decay=0.1,
    decay_type="linear",
):
    """Warmup, cosine, and wsd-like decay schedule.
    Args:
        n_iterations: total number of iterations
        n_warmup: number of warmup iterations
        anneal_end_factor: factor at which cosine annealing ends
        final_lr_factor: factor by which to reduce max_lr at the end
        init_div_factor: initial division factor for warmup
        fract_decay: fraction of iterations used for decay
        decay_type: type of decay after cosine phase
                   ['linear', 'exp', 'cosine', 'mirror_cosine', 'square', 'sqrt']
    Returns:
        schedule: a function that takes the current iteration and
        returns the multiplicative factor for the learning rate
    """
    valid_decay_types = ["linear", "exp", "cosine", "mirror_cosine", "square", "sqrt"]
    if decay_type not in valid_decay_types:
        raise ValueError(f"decay_type {decay_type} is not in {valid_decay_types}")

    max_lr = 1.0
    base_lr = max_lr / init_div_factor
    # final_lr = base_lr / final_lr_factor
    n_decay_steps = int(fract_decay * n_iterations)
    n_hold = n_iterations - n_decay_steps
    cosine_start = n_warmup
    cosine_end = n_warmup + n_hold

    def schedule(step):
        if step < n_warmup:
            # Warmup phase
            return (step / n_warmup) + (1 - step / n_warmup) / init_div_factor

        elif step < cosine_end:
            # Cosine regime
            t = (step - cosine_start) / (cosine_end - cosine_start)
            return anneal_end_factor + (max_lr - anneal_end_factor) * 0.5 * (
                1 + math.cos(math.pi * t)
            )

        elif step < n_iterations:
            # Decay regime
            progress = (step - cosine_end) / n_decay_steps

            if decay_type == "linear":
                return final_lr_factor + (anneal_end_factor - final_lr_factor) * (
                    1 - progress
                )

            elif decay_type == "exp":
                return final_lr_factor + (anneal_end_factor - final_lr_factor) * (
                    final_lr_factor ** (progress)
                )

            elif decay_type == "cosine":
                return final_lr_factor + (anneal_end_factor - final_lr_factor) * (
                    (1 + math.cos(math.pi * progress)) * 0.5
                )

            elif decay_type == "mirror_cosine":
                cosine_value = final_lr_factor + (
                    anneal_end_factor - final_lr_factor
                ) * ((1 + math.cos(math.pi * progress)) * 0.5)
                linear_value = final_lr_factor + (
                    anneal_end_factor - final_lr_factor
                ) * (1 - progress)
                return linear_value * 2 - cosine_value

            elif decay_type == "square":
                return final_lr_factor + (anneal_end_factor - final_lr_factor) * (
                    1 - progress**2
                )

            elif decay_type == "sqrt":
                return final_lr_factor + (anneal_end_factor - final_lr_factor) * (
                    1 - math.sqrt(progress)
                )

        else:
            return final_lr_factor

    return schedule


def dd_schedule(
    n_iterations,
    n_warmup,
    fract_fisrt_decay,
    max_lr,
    first_final_lr_factor=1e-2,
    second_final_lr_factor=0.0,
    div_factor=1e2,
    first_decay_type="cosine",
    second_decay_type="linear",
):
    """Warmup, cosine annealing, and linear decay schedule.
    Args:
        n_iterations: total number of iterations
        n_warmup: number of warmup iterations
        fract_fisrt_decay: fraction of iterations for the first decay phase
        max_lr: the mamimum value of learning rate during the training
        first_final_lr_factor: factor by which to reduce max_lr at the end of the first decay phase
        second_final_lr_factor: factor by which to reduce first_final_lr_factor at the end of the second decay phase
        div_factor: initial division factor for warmup
        first_decay_type: which decay approach to use during the fisrt decay phase
        second_decay_type: which decay approach to use during the second decay phase
    Returns:
        schedule: a function that takes the current iteration and
        returns the multiplicative factor for the learning rate
    """
    if fract_fisrt_decay > 1.0:
        raise ValueError(
            "Invalid fract_fisrt_decay value: {}".format(fract_fisrt_decay)
        )
    n_fisrt_decay = int(fract_fisrt_decay * n_iterations)

    def schedule(step):
        if step < n_warmup:
            return (step / n_warmup) + (1 - step / n_warmup) / div_factor
        elif step < n_warmup + n_fisrt_decay:
            if first_decay_type == "cosine":
                return first_final_lr_factor + 0.5 * (
                    max_lr - first_final_lr_factor
                ) * (1 + math.cos(math.pi * (step - n_warmup) / n_fisrt_decay))
            elif first_decay_type == "linear":
                return first_final_lr_factor + (max_lr - first_final_lr_factor) * (
                    1 - (step - n_warmup) / n_fisrt_decay
                )
            elif first_decay_type == "exp":
                return first_final_lr_factor ** ((step - n_warmup) / n_fisrt_decay)
            elif first_decay_type == "mirror_cosine":
                cosine_value = (
                    first_final_lr_factor
                    + (max_lr - first_final_lr_factor)
                    * (1 + math.cos(math.pi * (step - n_warmup) / n_fisrt_decay))
                    * 0.5
                )
                linear_value = first_final_lr_factor + (
                    max_lr - first_final_lr_factor
                ) * (1 - (step - n_warmup) / n_fisrt_decay)
                return linear_value * 2 - cosine_value
            elif first_decay_type == "square":
                return first_final_lr_factor + (max_lr - first_final_lr_factor) * (
                    1 - ((step - n_warmup) / n_fisrt_decay) ** 2
                )
            elif first_decay_type == "sqrt":
                return first_final_lr_factor + (max_lr - first_final_lr_factor) * (
                    1 - math.sqrt((step - n_warmup) / n_fisrt_decay)
                )
            else:
                raise ValueError(
                    f"decay type {first_decay_type} is not in ['cosine','miror_cosine','linear','exp']"
                )
        elif step < n_iterations:
            if second_decay_type == "linear":
                return second_final_lr_factor + (
                    first_final_lr_factor - second_final_lr_factor
                ) * (
                    1
                    - (step - n_warmup - n_fisrt_decay) / (n_iterations - n_fisrt_decay)
                )
            elif second_decay_type == "cosine":
                return second_final_lr_factor + 0.5 * (
                    first_final_lr_factor - second_final_lr_factor
                ) * (
                    1
                    + math.cos(
                        math.pi
                        * (step - n_warmup - n_fisrt_decay)
                        / (n_iterations - n_fisrt_decay)
                    )
                )
            elif second_decay_type == "exp":
                return first_final_lr_factor ** (
                    (step - n_warmup - n_fisrt_decay) / (n_iterations - n_fisrt_decay)
                )
            elif second_decay_type == "mirror_cosine":
                cosine_value = (
                    second_final_lr_factor
                    + (first_final_lr_factor - second_final_lr_factor)
                    * (
                        1
                        + math.cos(
                            math.pi
                            * (step - n_warmup - n_fisrt_decay)
                            / (n_iterations - n_fisrt_decay)
                        )
                    )
                    * 0.5
                )
                linear_value = second_final_lr_factor + (
                    first_final_lr_factor - second_final_lr_factor
                ) * (
                    1
                    - (step - n_warmup - n_fisrt_decay) / (n_iterations - n_fisrt_decay)
                )
                return linear_value * 2 - cosine_value
            elif second_decay_type == "square":
                return second_final_lr_factor + (
                    first_final_lr_factor - second_final_lr_factor
                ) * (
                    1
                    - (
                        (step - n_warmup - n_fisrt_decay)
                        / (n_iterations - n_fisrt_decay)
                    )
                    ** 2
                )
            elif second_decay_type == "sqrt":
                return second_final_lr_factor + (
                    first_final_lr_factor - second_final_lr_factor
                ) * (
                    1
                    - math.sqrt(
                        (step - n_warmup - n_fisrt_decay)
                        / (n_iterations - n_fisrt_decay)
                    )
                )
            else:
                raise ValueError(
                    f"decay type {second_decay_type} is not in ['cosine','miror_cosine','linear','exp']"
                )
        else:
            return second_final_lr_factor

    return schedule
