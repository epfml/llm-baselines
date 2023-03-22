from contextlib import nullcontext
import torch
import torch.nn.functional as F
import wandb
import time 
import copy

from .utils import eval, get_batch


def train_base(model, opt, data, scheduler, iterations, acc_steps, batch_size, sequence_length, eval_freq, ckpt_path, extra_args):

    device_type = 'cuda' if 'cuda' in str(extra_args.device) else 'cpu'
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)#extra_args.dtype)

    itr, substep, best_val_loss, text_table = 0, 0, float('inf'), None # best_val_loss not used atm, early stopping not recommended but possible 

    stats = {'train_loss': [], 'val_loss': [], 'val_pp': [], 'val_acc': []}

    num_substeps_per_epoch = len(data['train']) // (batch_size * sequence_length)
    
    if not extra_args.no_compile:
        print(f"Compiling model ...")
        model = torch.compile(model) # requires pytorch 2.0+

    model.train()

    t0 = time.time()
    while itr < iterations:
        for _ in range(acc_steps): # gradient accumulation
            x, y = get_batch(data['train'], sequence_length, batch_size, device=extra_args.device)
            print(x.shape)
            with ctx:
                outputs = model(x, targets=y)

            loss = outputs['loss']
            loss.backward()
            substep += 1

        opt.step()
        scheduler.step()
        opt.zero_grad(set_to_none=True)
        itr += 1

        if itr % eval_freq == 0 or itr == iterations: # eval
            t1 = time.time()
            dt = t1 - t0
            epoch = substep//num_substeps_per_epoch
            
            model.eval()
            train_loss = loss.detach().cpu().item()
            current_lr = scheduler.get_last_lr()[0] if scheduler is not None else extra_args.lr
            val_acc, val_loss, val_perplexity = eval(model, data['val'], sequence_length, batch_size, 
                                                     extra_args.device, max_num_batches=24, ctx=ctx)
            
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
                        text_table = wandb.Table(columns=["itr", "val-loss", "text"])

                    out_str = model.generate_from_string(extra_args.eval_seq_prefix, max_new_tokens=40, temperature=0.9, top_k=None)
                    text_table.add_data(itr, val_loss, out_str)
                    wandb.log({f"generated-text-{wandb.run.name}": copy.copy(text_table)}) # why a copy? see github.com/wandb/wandb/issues/2981

            model.train()
            t0 = time.time()

    checkpoint = {
        'model': model.state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'itr': itr,
    }
    print(f"saving checkpoint to {ckpt_path}")
    torch.save(checkpoint, ckpt_path)

    return stats

