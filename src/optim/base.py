from contextlib import nullcontext
from data.utils import get_dataloader

import torch
import torch.nn.functional as F
import wandb
import time 
import itertools
import copy
import random
import os
import numpy as np
from .utils import eval, get_batch, save_checkpoint


def train_base(model, opt, data, data_seed, scheduler, iterations, acc_steps, batch_size, sequence_length, eval_freq, ckpt_path, distributed_backend,extra_args, itr=0,rng_state_dict=None):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.bfloat16)  # extra_args.dtype)
    best_val_loss, text_table = float('inf'), None # best_val_loss not used atm, early stopping not recommended but possible 
    substep = itr * acc_steps
    data["train"], train_sampler = get_dataloader(
        data["train"],
        sequence_length=sequence_length,
        batch_size=batch_size,
        seed=data_seed,
        distributed_backend=distributed_backend,
    )
    
    data["val"], val_sampler = get_dataloader(
        data["val"],
        sequence_length=sequence_length,
        batch_size=batch_size,
        seed=data_seed,
    )

    num_substeps_per_epoch = len(data["train"])
    train_epochs = substep//num_substeps_per_epoch
    
    if rng_state_dict is not None and  rng_state_dict.get("train_sampler_state", None) is not None:
        train_sampler.generator.set_state(rng_state_dict["train_sampler_state"])
    if hasattr(train_sampler, "set_epoch"):
        train_sampler.set_epoch(train_epochs)
    else:
        sampler_state_before_iter = train_sampler.generator.get_state()
    data_train_iter = iter(data["train"])

    
    # for val data we don't care about epochs? just cycle through (no need to set_epoch to reshuffle)
    data_val_iter = itertools.cycle(data["val"])

    stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}

   
    
    if extra_args.compile:
        print(f"Compiling model ...")
        model = torch.compile(model) # requires pytorch 2.0+

    model.train()

    t0 = time.time()
    
    if rng_state_dict is not  None:
        torch.set_rng_state(rng_state_dict["cpu_rng_state"])
        torch.cuda.set_rng_state(rng_state_dict["gpu_rng_state"])
        np.random.set_state(rng_state_dict["numpy_rng_state"])
        random.setstate(rng_state_dict["py_rng_state"])
    for _ in range(substep % num_substeps_per_epoch):
        get_batch(data_train_iter, device=extra_args.device)

    
    while itr < iterations:
            
        for microstep_idx in range(acc_steps):  # gradient accumulation
            x, y = get_batch(data_train_iter, device=extra_args.device)
            
            with type_ctx:
                with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
                    outputs = model(x, targets=y)

            loss = outputs['loss'] / acc_steps
            loss.backward()
            substep += 1
            if substep % len(data["train"]) == 0:
                train_epochs += 1
                print(f"Train epoch {train_epochs} done (full pass over training data)")
                if hasattr(train_sampler, "set_epoch"):
                    # set epoch for reshuffling between epochs
                    train_sampler.set_epoch(train_epochs)
                    sampler_state_before_iter = None
                else:
                    sampler_state_before_iter = train_sampler.generator.get_state()
                data_train_iter = iter(data["train"])


        if extra_args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)
        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1

        if itr % eval_freq == 0 or itr == iterations: # from here it's only evaluation code, all the training is above
            if distributed_backend.is_master_process():
                t1 = time.time()
                dt = t1 - t0
                epoch = substep//num_substeps_per_epoch

                model.eval()
                train_loss = loss.detach().cpu().item() * acc_steps
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
                
                eval_steps = 24  # or choose full val set?
                val_acc, val_loss, val_perplexity = eval(
                    model,
                    data_val_iter,
                    extra_args.device,
                    max_num_batches=eval_steps,
                    ctx=type_ctx,
                )

                print_string = f"{epoch}/{itr} [train] loss={train_loss:.3f} [val] loss={val_loss:.3f}, pp={val_perplexity:.2f}, acc={val_acc:3f}"
                print_string += f" [time per itr] {dt*1000/eval_freq:.2f}ms"
                if scheduler is not None:
                    print_string += f" [lr] {current_lr:.5f}"
                print(print_string)

                if extra_args.wandb:
                    wandb.log({
                        "iter": itr,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/perplexity": val_perplexity,
                        "val/acc": val_acc,
                        "lr": current_lr,
                    })

                    if extra_args.eval_seq_prefix != 'none' and (itr % (eval_freq * 5) == 0 or itr == iterations):
                        if text_table is None:
                            text_table = wandb.Table(columns=["itr", "val-pp", "text"])

                        out_str = distributed_backend.get_raw_model(model).generate_from_string(
                            extra_args.eval_seq_prefix, max_new_tokens=40, temperature=0.9, top_k=None)
                        text_table.add_data(itr, val_perplexity, out_str)
                        # why a copy? see github.com/wandb/wandb/issues/2981
                        wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)})

                model.train()
                t0 = time.time()
        if distributed_backend.is_master_process():
            if extra_args.save_checkpoint_freq is not None and itr % extra_args.save_checkpoint_freq == 0:
                print(f"saving checkpoint to {os.path.dirname(ckpt_path)}/ckpt_{itr}.pt")
                save_checkpoint(distributed_backend=distributed_backend,
                                model=model,
                                opt=opt,
                                scheduler=scheduler,
                                itr=itr,
                                cpu_rng_state=torch.get_rng_state(),
                                gpu_rng_state=torch.cuda.get_rng_state(),
                                numpy_rng_state=np.random.get_state(),
                                py_rng_state=random.getstate(),
                                train_sampler_state=sampler_state_before_iter,
                                ckpt_path=os.path.join(os.path.dirname(ckpt_path), f"ckpt_{itr}.pt"))
                
    if distributed_backend.is_master_process():
        print(f"saving checkpoint to {ckpt_path}")
        save_checkpoint(distributed_backend=distributed_backend,
                        model=model,
                        opt=opt,
                        scheduler=scheduler,
                        itr=itr,
                        ckpt_path=ckpt_path)

    return stats
