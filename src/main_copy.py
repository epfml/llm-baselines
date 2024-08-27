import os
import sys
import numpy as np
import torch
import inspect
import json
import copy
import argparse
import random
import wandb

import config
from models.utils import get_model, get_model_and_tokenizer
from data.utils import get_dataset, get_shakespeare, token2seq
from optim.base_copy import train_base
import distributed

import pdb
def get_args():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--config_format', default='base', choices=config.registered_formats())

    args, rem_args = parser.parse_known_args()

    return config.parse_args_with_format(format=args.config_format, base_parser=parser, args=rem_args, namespace=args)


def main(args): 
    ## DEVICE SETUP
    torch.backends.cuda.matmul.allow_tf32 = True # allows us to make sure we're able to use tensorfloat32 during training
    torch.backends.cudnn.allow_tf32 = True

    distributed_backend = distributed.make_backend_from_args(args)
    args = distributed_backend.get_adjusted_args_for_process(args)

    args.device = torch.device(args.device)
    device_type = "cuda" if "cuda" in str(args.device) else "cpu"
    if device_type == "cuda":
        torch.cuda.set_device(args.device)

    ## RANDOM SEEDS
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    ## MODEL AND TOKENIZER SETUP
    model, tokenizer = get_model_and_tokenizer(args)
    model = model.to(args.device)
    model = distributed_backend.transform_model(model)
    model_paras = model.parameters()
    voab_size = tokenizer.vocab_size
    
    ## DATA SETUP
    print(f"Loading dataset '{args.dataset}'")
    if args.dataset == 'shakespeare':
        train_tokens, val_tokens = get_shakespeare(tokenizer = tokenizer) # data is a dict: {'train': a list of tokenized seqs, 'val': val_tokenized}
    else:
        raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    train_seq = token2seq(tokens = train_tokens, max_seq_length = args.sequence_length)
    val_seq = token2seq(tokens = val_tokens, max_seq_length = args.sequence_length)
    # random generate some data added to the training data
    np.random.seed(args.data_rd_seed)
    random_tokens = np.random.randint(low=0, high=voab_size-1, size=(args.num_rand_tok,), dtype=np.uint16)
    random_seq = token2seq(tokens = random_tokens, max_seq_length = args.sequence_length)
    train_seq += random_seq
    data = {'train': train_seq, 'val': val_seq}

    print(f"Num training seqs: {len(data['train'])}, including {len(random_seq)} random seqs")
    print(f"Num validation seqs: {len(data['val'])}")

    ## OPTIMIZER SETUP
    if args.opt == 'adamw':
        use_fused = (device_type == 'cuda') and ('fused' in inspect.signature(torch.optim.AdamW).parameters)
        print(f"using fused AdamW: {use_fused}")
        extra_args = dict(fused=True) if use_fused else dict()
        opt = torch.optim.AdamW(model_paras, lr=args.lr, betas=(args.beta1, args.beta2),
                                weight_decay=args.weight_decay, **extra_args)
    else:
        opt = torch.optim.SGD(model_paras, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    
    if args.scheduler != 'none':
        if args.scheduler in ['cos', 'linear']:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=opt, max_lr=args.lr, 
                                                            total_steps=args.iterations, 
                                                            pct_start=args.warmup_percent, 
                                                            anneal_strategy=args.scheduler, 
                                                            cycle_momentum=False, div_factor=1e2, final_div_factor=.1)
        else:
            raise NotImplementedError(f"Unknown scheduler type: {args.scheduler}.")
    else:
        scheduler = None

    args.world_size = distributed_backend.get_world_size()
    print(f"world_size: {args.world_size}")

    ## LOGGING SETUP
    exp_name = args.exp_name
    if distributed_backend.is_master_process() and args.wandb:
        params_copy = copy.deepcopy(vars(args))
        del params_copy['device']
        wandb.init(project=args.wandb_project, name=exp_name, config=params_copy)
    
    ckpt_path = os.path.join(args.results_base_folder, args.dataset, args.model, exp_name)
    if not os.path.exists(ckpt_path):
        if distributed_backend.is_master_process():
            os.makedirs(ckpt_path)
        distributed_backend.sync()
    elif os.path.isfile(os.path.join(ckpt_path, "summary.json")): # the experiment was already completed
        print(f"Already found experiment '{ckpt_path}'.\nSkipping.")
        sys.exit(0)

    ## TRAINING SETUP
    itr = 0
    rng_state_dict = None

    train = train_base

    print(f"\nTraining model={args.model} \n{vars(args)}\n")
    stats = train(model, opt, data, args.gamma, args.num_curated_batch, args.num_rand_tok, 
                  args.data_seed, scheduler, args.iterations, args.acc_steps, args.batch_size, 
                  args.sequence_length, eval_freq=args.eval_freq, 
                  distributed_backend=distributed_backend,
                  ckpt_path=f"{ckpt_path}/ckpt.pt", itr=itr, 
                  rng_state_dict=rng_state_dict, extra_args=args)
    
    args.device = None
    args.dtype = None
    stats['args'] = vars(args)
    if distributed_backend.is_master_process():
        with open(f"{ckpt_path}/summary.json", "w") as fs:
            json.dump(stats, fs)
    distributed_backend.finalize()


if __name__ == "__main__":
    args = get_args()
    main(args)
