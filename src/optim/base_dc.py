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

from torch import mean
import pdb

def train_base_dc(model, opt, data, gamma, num_curated_tok, num_rand_tok, data_seed, scheduler, iterations, acc_steps, batch_size, sequence_length, 
               eval_freq, ckpt_path, distributed_backend,extra_args, itr=0,rng_state_dict=None):
    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    type_ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(
        device_type=device_type, dtype=torch.bfloat16)  # extra_args.dtype
    best_val_loss, text_table = float('inf'), None # best_val_loss not used atm, early stopping not recommended but possible 
    substep = itr * acc_steps
    
    data_cnt = 0
    
    print(f'Num curated tokens: {num_curated_tok}')
    data["curated"], curated_sampler = get_dataloader(
        data["train"][:num_curated_tok],
        sequence_length=sequence_length,
        batch_size=batch_size,
        seed=data_seed,
    )

    num_curate_batch = len(data["curated"])

    num_train_seq = int(np.ceil(len(data["train"][num_curated_tok :]) / sequence_length))
    num_rand_seq = int(np.ceil(num_rand_tok / sequence_length))
    w = torch.ones(num_train_seq, device=device_type)
    w_gt = np.zeros(num_train_seq)
    w_gt[num_train_seq - num_rand_seq:] = 1
    w_gt_sum = sum(w_gt)

    w_loader, w_sampler = get_dataloader(
        w_gt,
        sequence_length=1,
        batch_size=1,
        seed=data_seed,
        distributed_backend=distributed_backend,
    )
    # print(f'w_loader: {w_loader}')

    w_list1 = []
    w_list2 = []
    for wi, wj in w_loader:
        w_list1.append(wi[0])
        w_list2.append(wj[0])
    w_list1.append(w_list2[-1])
    w_gt_tensor = torch.tensor(w_list1, device=device_type)
    w_gt = w_gt_tensor.bool()
    print(f'w_gt: {w_gt}')
    # pdb.set_trace()

    num_clean_seq  = num_train_seq - num_rand_seq
    print(f'Num clean seq in train: {num_clean_seq}')
    print(f'Num random seq: {num_rand_seq}')

    w_gap = mean(w[w_gt]) - mean(w[~w_gt]) 
    w_error = mean((w - w_gt_tensor) ** 2)
    print(f'Initial w_gap: {w_gap}')
    print(f'Initial w_error: {w_error}')
    
    # pdb.set_trace()

    data["train"], train_sampler = get_dataloader(
        data["train"][num_curated_tok :],
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
    train_epochs = substep//num_substeps_per_epoch # counting the current epoch
    
    if rng_state_dict is not None and  rng_state_dict.get("train_sampler_state", None) is not None:
        train_sampler.generator.set_state(rng_state_dict["train_sampler_state"])
    if hasattr(train_sampler, "set_epoch"):
        train_sampler.set_epoch(train_epochs)
    else:
        sampler_state_before_iter = train_sampler.generator.get_state()
    data_train_iter = iter(data["train"])

    print(f'Data train: {len(data["train"])}')
    print(f'Data eval: {len(data["val"])}')

    # for val data we don't care about epochs? just cycle through (no need to set_epoch to reshuffle)
    data_val_iter = itertools.cycle(data["val"])
    data_curated_iter = itertools.cycle(data["curated"])

    stats = {"curated_loss": [], "train_loss": [], "val_loss": [], "val_pp": [], "val_acc": []}

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
        for microstep_idx in range(acc_steps):
            grad0 = {name: torch.zeros_like(param) for name, param in model.named_parameters()}
            loss0 = 0
            for _ in range(num_curate_batch):
                x0, y0 = get_batch(data_curated_iter, device=extra_args.device)
                with type_ctx:
                    with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
                        outputs = model(x0, targets=y0)
                loss0i = outputs['loss'] / acc_steps
                opt.zero_grad(set_to_none=True)
                loss0i.backward()
                grad0i = {name: param.grad.clone() for name, param in model.named_parameters()}
                for name, param in model.named_parameters():
                    grad0[name] += grad0i[name] / num_curate_batch
                loss0 += loss0i / num_curate_batch

            grads_batch = copy.deepcopy(grad0)
            loss = loss0.clone()

            x, y = get_batch(data_train_iter, device=extra_args.device)

            # # to check the batch grad is the MEAN (not the sum) of the individual grads
            # with type_ctx:
            #     with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
            #         output = model(x, targets=y)  
            # loss = output['loss']
            # opt.zero_grad(set_to_none=True)
            # loss.backward()
            # grad_b = {name: param.grad.clone() for name, param in model.named_parameters()}
            # print('Grad_b: ', grad_b['transformer.h.22.mlp.c_proj.weight'][:2])
            
            # grad_sum = {name: torch.zeros_like(grad_b[name]) for name in grad_b.keys()}
            # for i in range(x.size(0)):  # Iterate over each data point in the batch
            #     xi = x[i].unsqueeze(0)  # Get the i-th data point
            #     yi = y[i].unsqueeze(0)  # Get the i-th target
            #     with type_ctx:
            #         with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
            #             output_i = model(xi, targets=yi)

            #     lossi = output_i['loss']
            #     opt.zero_grad(set_to_none=True)
            #     lossi.backward()
            #     gradi = {name: param.grad.clone() for name, param in model.named_parameters()}
            #     for name, param in model.named_parameters():
            #         grad_sum[name] += gradi[name]
            # print('Grad_sum: ', grad_sum['transformer.h.22.mlp.c_proj.weight'][:2])

            # pdb.set_trace()
                

            

            for i in range(x.size(0)):  # Iterate over each data point in the batch
                if w[data_cnt] > 1e-2:
                    xi = x[i].unsqueeze(0)  # Get the i-th data point
                    yi = y[i].unsqueeze(0)  # Get the i-th target
                    with type_ctx:
                        with distributed_backend.get_context_for_microstep_forward(model=model, microstep_idx=microstep_idx, gradient_accumulation_steps=acc_steps):
                            output_i = model(xi, targets=yi)
                            
                    lossi = output_i['loss']
                    loss += w[data_cnt] * lossi / batch_size
                    # 
                    # lossi = w[data_cnt] * output_i['loss']
                    # loss += lossi  

                    opt.zero_grad(set_to_none=True)
                    lossi.backward()
                    gradi = {name: param.grad.clone() for name, param in model.named_parameters()}

                    # inner_product = sum((torch.flatten(grad0[name]) * torch.flatten(gradi[name])).sum() for name in grad0.keys())
                    # print(inner_product)
                    # w[data_cnt] += gamma * inner_product

                    # wj = wj + ||g0|| cos_sim
                    cos_sim_g0 = sum((torch.flatten(grad0[name]) * torch.flatten(gradi[name])).sum() for name in grad0.keys()) /  torch.norm(torch.cat([torch.flatten(gradi[name]) for name in grad0.keys()]))
                    
                    w[data_cnt] += gamma * cos_sim_g0
                    w[data_cnt] = torch.clamp(w[i], 0, 1)
                    
                    # Accumulate the gradients
                    for name, param in model.named_parameters():
                        if param.grad is not None:
                            grads_batch[name] += w[data_cnt] * gradi[name] / batch_size

                data_cnt += 1
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param.grad = grads_batch[name]
       
        if extra_args.grad_clip != 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), extra_args.grad_clip)

        opt.step()
        scheduler.step()
        itr += 1
        substep += 1

        if data_cnt % (num_train_seq-1) == 0:
            data_cnt = 0

        if substep % len(data["train"]) == 0:
            if sum(w) <= w_gt_sum:
                w[w > 0.5] = 1
                w[w <= 0.5] = 0
            train_epochs += 1
            print(f"Train epoch {train_epochs} done (full pass over training data)")
            if hasattr(train_sampler, "set_epoch"):
                # set epoch for reshuffling between epochs
                train_sampler.set_epoch(train_epochs)
                sampler_state_before_iter = None
            else:
                sampler_state_before_iter = train_sampler.generator.get_state()
                
            data_train_iter = iter(data["train"])

        # data_curated_iter = iter(data["curated"])

        # if itr % eval_freq == 0 or itr == iterations: # from here it's only evaluation code, all the training is above
        if substep % len(data["train"]) == 0 or itr == iterations: # when finish one epoch, do evaluation
            if distributed_backend.is_master_process():
                t1 = time.time()
                dt = t1 - t0
                epoch = substep//num_substeps_per_epoch

                model.eval()
                curated_loss = loss0.detach().cpu().item()
                train_loss = loss.detach().cpu().item() * acc_steps
                current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
                
                eval_steps = (
                    24 if itr < iterations else len(data["val"])
                )
                val_acc, val_loss, val_perplexity = eval(
                    model,
                    data_val_iter,
                    extra_args.device,
                    max_num_batches=eval_steps,
                    ctx=type_ctx,
                )
                
                w_gap = mean(w[w_gt]) - mean(w[~w_gt])    

                print_string = f"{epoch}/{itr} [curated] loss={curated_loss:.3f} w_sum={sum(w):.3f} w_gap={w_gap:.3f} [train] loss={train_loss:.3f} [val] loss={val_loss:.3f}, pp={val_perplexity:.2f}, acc={val_acc:3f}"
                print_string += f" [time per itr] {dt*1000/eval_freq:.2f}ms"
                # print_string += f" w={w}"
                if scheduler is not None:
                    print_string += f" [lr] {current_lr:.5f}"
                print(print_string)

                if extra_args.wandb:
                    logs = {
                        "iter": itr,
                        "w_sum": sum(w),
                        "curated/loss": curated_loss,
                        "train/loss": train_loss,
                        "val/loss": val_loss,
                        "val/perplexity": val_perplexity,
                        "val/acc": val_acc,
                        "lr": current_lr,
                        "w_gap": w_gap,
                        "w1_mean": mean(w[w_gt]),
                        "w0_mean": mean(w[~w_gt]),
                        "w": w,
                        "w_error": mean((w - w_gt_tensor) ** 2),
                    }

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
