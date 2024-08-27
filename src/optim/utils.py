import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext, contextmanager, ExitStack


def get_batch(dataloader, device="cpu"):
    x, y = next(dataloader)
    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)
    return x, y

def get_one_batch(dataloader, device="cpu"):
    x = next(dataloader)
    if "cuda" in torch.device(device).type:
        x = x.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
    return x

@torch.no_grad()
def eval(model, data_val_iter, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    loss_list_val, acc_list = [], []

    for _ in range(max_num_batches): 
        x, y = get_batch(data_val_iter, device=device)
        with ctx:
            outputs = model(x, targets=y, get_logits=True)
        val_loss = outputs['loss']
        loss_list_val.append(val_loss)
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity

@torch.no_grad()
def eval_gpt2(model, data_val_iter, device='cpu', max_num_batches=24, ctx=nullcontext()):
    assert model.training == False

    loss_list, acc_list = [], []

    for _ in range(max_num_batches): 
        x = get_one_batch(data_val_iter, device=device)
        with ctx:
            outputs = model(x, labels=x)
        val_loss = outputs.loss.item()
        loss_list.append(val_loss)
        # Calculate token-level accuracy
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = x[..., 1:].contiguous()
        # Flatten the tokens
        # loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # print(f'logits shape: {logits.shape}')
        predictions = torch.argmax(shift_logits, dim=-1)
        # print(f'predictions shape: {predictions.shape}')
        acc = (predictions == shift_labels).float().mean()
        acc_list.append(acc)

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity

def save_checkpoint(distributed_backend, model, opt, scheduler, itr, ckpt_path, **extra_args):

    checkpoint = dict({
        'model': distributed_backend.get_raw_model(model).state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'itr': itr,
    }, **extra_args)

    torch.save(checkpoint, ckpt_path)
