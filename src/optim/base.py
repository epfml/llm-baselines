import copy
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

import wandb

# from logger.logger import DynamicsLogger
from .utils import (eval, get_batch, load_checkpoint, load_worker_state,
                    save_checkpoint, save_worker_state)


def train(
    model,
    opt,
    datareaders,
    scheduler,
    exp_dir,
    distributed_backend,
    cfg,
):
    not_compiled_model = model
    if cfg.compile:
        print(f"Compiling model ...")
        model = torch.compile(model)

    if "cuda" in cfg.device:
        type_ctx = torch.amp.autocast(
            device_type="cuda",
            dtype={
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }[cfg.dtype],
        )
    else:
        type_ctx = nullcontext()

    if cfg.resume_from:
        # This is a full resume including the model weights, optimizer, state
        # dataloader state, random seed, etc. Not indended for fine tuning or
        # other scenarios where some of these should change.
        print(f"\nResuming Training From {cfg.resume_from}")
        ckpt_dir = Path(cfg.resume_from)
        curr_iter = load_checkpoint(
            model,
            opt,
            scheduler,
            ckpt_dir / "main.pt",
            cfg.device,
        )
        load_worker_state(ckpt_dir)
    else:
        curr_iter = 0

    # if distributed_backend.is_master_process() and cfg.log_dynamics:
    #     with open(cfg.dynamics_logger_cfg, "r") as f:
    #         dlcfg = yaml.safe_load(f)

    #     # Hooks into optimizer
    #     dlogger = DynamicsLogger(
    #         model, opt, dlcfg, cfg.results_base_folder, wandb=cfg.wandb
    #     )
    #     dlogger.iteration = curr_iter

    substep = curr_iter * cfg.acc_steps
    train_reader, val_reader = datareaders["train"], datareaders["val"]
    train_reader.set_step(substep)
    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
    grad_norms = []
    model.train()

    while curr_iter <= cfg.iterations:
        # Save permanent checkpoint
        if cfg.permanent_ckpt_interval > 0:
            if curr_iter % cfg.permanent_ckpt_interval == 0:
                ckpt_dir = exp_dir / "ckpts" / str(curr_iter)
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir)

        # Save temporary checkpoint for resuming training
        if cfg.latest_ckpt_interval > 0:
            if curr_iter % cfg.latest_ckpt_interval == 0 or curr_iter == cfg.iterations:
                ckpt_dir = exp_dir / "ckpts" / "latest"
                if distributed_backend.is_master_process():
                    save_checkpoint(model, opt, scheduler, curr_iter, ckpt_dir)
                save_worker_state(ckpt_dir)

        ws = distributed_backend.get_world_size()
        tokens = ws * substep * cfg.sequence_length * cfg.batch_size
        epoch = tokens / train_reader.num_tokens
        if (
            curr_iter % cfg.eval_interval == 0
            or curr_iter == cfg.iterations
            or (curr_iter in cfg.full_eval_at)
        ):
            eval_and_log(
                tokens,
                curr_iter,
                epoch,
                model,
                val_reader,
                type_ctx,
                distributed_backend,
                cfg,
                opt,
                full_eval=(curr_iter in cfg.full_eval_at),
            )

        if curr_iter == cfg.iterations:
            # Save checkpoints and evaluate at final iteration, but no need to train further
            break

        # Train model
        t_start = time.perf_counter_ns()
        for microstep_idx in range(cfg.acc_steps):  # gradient accumulation
            x, y = get_batch(train_reader, device=cfg.device)
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(
                    model=model,
                    microstep_idx=microstep_idx,
                    gradient_accumulation_steps=cfg.acc_steps,
                ):
                    outputs = model(x, targets=y)

            loss = outputs["loss"] / cfg.acc_steps
            loss.backward()
            substep += 1

        if cfg.grad_clip != 0.0:
            if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.module.parameters(), cfg.grad_clip
                )
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), cfg.grad_clip
                )
            grad_norms.append(grad_norm)

        if cfg.opt == "sf-sgd" or cfg.opt == "sf-adamw":
            opt.train()
        (
            opt.step()
            if cfg.opt != "sophiag"
            else opt.step(bs=cfg.sophia_bs * cfg.sequence_length)
        )
        if cfg.scheduler != "none":
            scheduler.step()
        if cfg.opt == "sophiag":
            opt.zero_grad(set_to_none=True)
            if curr_iter % 10 == 10 - 1:
                sample_again = model(x, targets=y, get_logits=True)
                samp_dist = torch.distributions.Categorical(
                    logits=sample_again["logits"]
                )
                y_sample = samp_dist.sample()
                loss_sampled = torch.nn.functional.cross_entropy(
                    sample_again["logits"].view(-1, sample_again["logits"].size(-1)),
                    y_sample.view(-1),
                    ignore_index=-1,
                )
                (loss_sampled / cfg.acc_steps).backward()
                opt.update_hessian()
                opt.zero_grad(set_to_none=True)
                model.zero_grad()
        elif cfg.opt == "mars":
            opt.zero_grad(set_to_none=True)
            opt.update_last_grad()
        else:
            opt.zero_grad(set_to_none=True)
        # opt.zero_grad(set_to_none=True)
        dt = (time.perf_counter_ns() - t_start) / 1e9

        curr_iter += 1

        if (
            cfg.log_interval
            and curr_iter % cfg.log_interval == 0
            and distributed_backend.is_master_process()  # Only log on master rank
        ):
            train_loss = loss.detach().cpu().item() * cfg.acc_steps

            current_lrs = [param_group["lr"] for param_group in opt.param_groups]

            print(
                f"Train: Iter={curr_iter} ({epoch:0.3f} epochs) "
                f"train_loss={train_loss:.3f} iter_dt={dt:.2e}s "
                f"lr={current_lrs[0]:.2e}"
            )

            if cfg.wandb:
                wandb.log(
                    {
                        "tokens": tokens,
                        "iter": curr_iter,
                        "train/loss": train_loss,
                        "train/perplexity": 2.71828**train_loss,
                        "lr": current_lrs[0],
                        "iter_dt": dt,
                        "max_grad_norm": max(grad_norms).item() if grad_norms else 0,
                        "mean_grad_norm": (
                            torch.tensor(grad_norms).mean().item() if grad_norms else 0
                        ),
                    }
                )

            grad_norms = []

    return stats


def eval_and_log(
    tokens,
    curr_iter,
    epoch,
    model,
    val_reader,
    type_ctx,
    distributed_backend,
    cfg,
    opt,
    full_eval=False,
):
    if not distributed_backend.is_master_process():
        # Only evaluate and log on master rank
        return

    model.eval()
    if cfg.opt == "sf-sgd" or cfg.opt == "sf-adamw":
        opt.eval()

    if curr_iter == cfg.iterations or full_eval:
        max_num_batches = val_reader.num_batches()
    else:
        max_num_batches = cfg.eval_batches

    # to make sure we start from the beginning of the validation set,
    # i.e. repeat the same batches
    val_reader.set_step(0)
    val_acc, val_loss, val_perplexity = eval(
        model,
        val_reader,
        cfg.device,
        max_num_batches=max_num_batches,
        ctx=type_ctx,
        cfg=cfg,
    )

    print(
        f">Eval: Iter={curr_iter} ({epoch:0.3f} epochs) "
        f"val_loss={val_loss:.3f} "
        f"val_pp={val_perplexity:.3f} "
        f"val_acc={val_acc:3f}"
    )

    if cfg.wandb:
        if curr_iter == cfg.iterations or full_eval:
            logs = {
                "tokens": tokens,
                "iter": curr_iter,
                "final-val/loss": val_loss,
                "final-val/perplexity": val_perplexity,
                "final-val/acc": val_acc,
            }
        else:
            logs = {
                "tokens": tokens,
                "iter": curr_iter,
                "val/loss": val_loss,
                "val/perplexity": val_perplexity,
                "val/acc": val_acc,
            }

        wandb.log(logs)
        if cfg.eval_seq_prefix != "none" and (
            curr_iter % (cfg.eval_interval * 5) == 0 or curr_iter == cfg.iterations
        ):
            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

            out_str = distributed_backend.get_raw_model(model).generate_from_string(
                cfg.eval_seq_prefix,
                max_new_tokens=40,
                temperature=0.9,
                top_k=None,
            )
            text_table.add_data(curr_iter, val_perplexity, out_str)
            # why a copy? see github.com/wandb/wandb/issues/2981
            wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})
    model.train()
