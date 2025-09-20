#!/bin/bash

torchrun --nproc_per_node=1 ./src/main.py --config_format base --model llama --distributed_backend nccl \
    --n_embd 768 --n_head 12 --n_layer 12 \
    --batch_size 64 --sequence_length 512 --acc_steps 4 \
    --dataset slimpajama --iterations 128000 \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
    --opt ademamix --lr 1e-3 --weight_decay 0.1 --scheduler cos \
    --beta1 0.9 --beta2 0.999 \
    --adema_beta3 0.999 --adema_alpha 8.0 \
    --adema_beta3_warmup 128000 --adema_alpha_warmup 128000 \
       --wandb --wandb_project "$YOUR_WANDB_PROJECT"  --wandb_entity "$YOUR_WANDB_ENTITY" \
    --eval_interval 115 --latest_ckpt_interval 1000 \