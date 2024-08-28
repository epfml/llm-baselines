from contextlib import nullcontext
from data.utils import get_dataloader, get_loader
from optim.utils import get_one_batch

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

from torch import mean
import pdb

def train_base(model, opt, data, gamma, num_curated_seqs, num_rand_seq, 
               data_seed, scheduler, iterations, acc_steps, batch_size, sequence_length, 
               eval_freq, ckpt_path, distributed_backend, extra_args, itr=0, rng_state_dict=None):
    ## DEVICE and CONTEXT
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.bfloat16)  # extra_args.dtype
    best_val_loss, text_table = float('inf'), None # best_val_loss not used atm, early stopping not recommended but possible 
    substep = itr * acc_steps

    ## DATALOADER
    data_loader = get_loader(data, batch_size,
                            distributed_backend=distributed_backend,
                            seed=data_seed) 
    num_batches_per_epoch = len(data_loader['train']) 
    num_curated_batches = len(data_loader['curated'])
    num_val_batches = len(data_loader['val'])
    train_sampler = data_loader['train_sampler']
    train_epochs = substep//num_batches_per_epoch # counting the current epoch

    data_train_iter = iter(data_loader['train'])
    # for val data we don't care about epochs? just cycle through (no need to set_epoch to reshuffle)
    data_val_iter = itertools.cycle(data_loader["val"])
    num_train_seq = len(data["train"])

    if data_loader['curated'] is not None:
        stats = {"curated_loss": [], "train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
        print(f'Num clean seq in train: {num_train_seq}')
        print(f'Num random seq: {num_rand_seq}')
        w = torch.ones(num_train_seq, device=device_type)
        w_gt = np.zeros(num_train_seq)
        w_gt[num_train_seq - num_rand_seq:] = 1
        w_gt = w_gt[data_loader['perm']]
        w_gt = torch.tensor(w_gt).bool()
        w_gt_sum = sum(w_gt)
        w_gap = mean(w[w_gt]) - mean(w[~w_gt]) 
        w_error = mean((w - w_gt) ** 2)
        print(f'Initial w_gap: {w_gap}')
        print(f'Initial w_error: {w_error}')
        data_cnt = 0
        dc_flag = True
    else:
        stats = {"train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}
        dc_flag = False

    if extra_args.compile:
        print(f"Compiling model ...")
        model = torch.compile(model) # requires pytorch 2.0+
    model.train()
    t0 = time.time()
    
    if rng_state_dict is not None:
        torch.set_rng_state(rng_state_dict["cpu_rng_state"])
        torch.cuda.set_rng_state(rng_state_dict["gpu_rng_state"])
        np.random.set_state(rng_state_dict["numpy_rng_state"])
        random.setstate(rng_state_dict["py_rng_state"])
    for _ in range(substep % num_batches_per_epoch):
        get_one_batch(data_train_iter, device=extra_args.device)

    while itr < iterations:
        if dc_flag:
            grad0 = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
            loss0 = 0
            for x0 in data_loader['curated']: # for each batch of curated data
                with type_ctx:
                    with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=0, gradient_accumulation_steps=acc_steps):
                        outputs = model(x0, targets=x0)
                loss0i = outputs['loss']
                opt.zero_grad(set_to_none=True)
                loss0i.backward()
                grad0i = {name: param.grad.clone() for name, param in model.named_parameters()}
                for name, param in model.named_parameters():
                    grad0[name] += grad0i[name] / num_curated_batches
                loss0 += loss0i.detach().cpu().item() / num_curated_batches
            grads_batch = copy.deepcopy(grad0)
            loss = loss0.clone()
        for microstep_idx in range(acc_steps):  # gradient accumulation
            x = get_one_batch(data_train_iter, device=extra_args.device)
            if dc_flag:
                for i in range(x.size(0)):
                    if w[data_cnt] > 1e-2:
                        xi = x[i].unsqueeze(0)
                        with type_ctx:
                            with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
                                outputs = model(xi, targets=xi)
                        lossi = outputs['loss']
                        loss += lossi * w[data_cnt] / batch_size
                        opt.zero_grad(set_to_none=True)
                        lossi.backward()
                        gradi = {name: param.grad.clone() for name, param in model.named_parameters()}
                        # wj = wj + ||g0|| cos_sim
                        cos_sim_g0 = sum((torch.flatten(grad0[name]) * torch.flatten(gradi[name])).sum() for name in grad0.keys()) /  torch.norm(torch.cat([torch.flatten(gradi[name]) for name in grad0.keys()]))
                        w[data_cnt] += gamma * cos_sim_g0
                        w[data_cnt] = torch.clamp(w[i], 0, 1)
                        for name, param in model.named_parameters():
                            if param.grad is not None:
                                grads_batch[name] += w[data_cnt] * gradi[name] / batch_size
                    data_cnt += 1
                for name, param in model.named_parameters():
                    if param.grad is not None:
                        param.grad = grads_batch[name]
                if data_cnt % (num_train_seq-1) == 0:
                    data_cnt = 0
            else: # no curated data
                with type_ctx:
                    with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
                        outputs = model(x, targets=x)
                loss = outputs['loss'] / acc_steps
                opt.zero_grad(set_to_none=True)
                loss.backward()
            
            substep += 1
            if substep % num_batches_per_epoch == 0:
                train_epochs += 1
                print(f"Train epoch {train_epochs} done (full pass over training data)")
                if hasattr(train_sampler, "set_epoch"):
                    # set epoch for reshuffling between epochs
                    train_sampler.set_epoch(train_epochs)
                    sampler_state_before_iter = None
                data_train_iter = iter(data_loader["train"])
       
        if extra_args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)

        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1

        # if itr % eval_freq == 0 or itr == iterations: # from here it's only evaluation code, all the training is above
        if substep % num_batches_per_epoch == 0 or itr == iterations: # when finish one epoch, do evaluation
            if distributed_backend.is_master_process():
                t1 = time.time()
                dt = t1 - t0
                epoch = substep//num_batches_per_epoch

                model.eval()
                train_loss = loss.detach().cpu().item() * acc_steps
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
                
                eval_steps = (
                    min(24, num_val_batches) if itr < iterations else num_val_batches
                )
                val_acc, val_loss, val_perplexity = eval(
                    model,
                    data_val_iter,
                    extra_args.device,
                    max_num_batches=eval_steps,
                    ctx=type_ctx,
                )
                
                print_string = f"{epoch}/{itr} [train] loss={train_loss:.3f} [val] loss={val_loss:.3f}, pp={val_perplexity:.2f}, acc={val_acc:3f}"
                if dc_flag:
                    curated_loss = loss0.detach().cpu().item()
                    w_gap = mean(w[w_gt]) - mean(w[~w_gt]) 
                    w_error = mean((w - w_gt) ** 2)
                    print_string += f" [curated] loss={curated_loss:.3f}, w_gap={w_gap:.3f}, w_error={w_error:.3f}, w_sum={sum(w):.3f}"
                print_string += f" [time per itr] {dt*1000/eval_freq:.2f}ms"
                if scheduler is not None:
                    print_string += f" [lr] {current_lr:.5f}"
                print(print_string)

                if extra_args.wandb:
                    logs = {
                        "iter": itr,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/perplexity": val_perplexity,
                        "val/acc": val_acc,
                        "lr": current_lr,
                    }
                    if dc_flag:
                        logs["curated/loss"] = curated_loss
                        logs["curated/w_gap"] = w_gap
                        logs["curated/w_error"] = w_error
                        logs["curated/w_sum"] = sum(w)
                        logs["curated/w1mean"] = mean(w[w_gt])
                        logs["curated/w0mean"] = mean(w[~w_gt])

                    if itr == iterations:
                        logs["val/final-ppl"] = val_perplexity
                        logs["val/final-acc"] = val_acc
                        logs["val/final-loss"] = val_loss

                    wandb.log(logs)

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
        
    artifact = wandb.Artifact('model_checkpoint', type='model')
    artifact.add_file(ckpt_path)
    wandb.log_artifact(artifact)

    return stats
