#!/bin/bash

torchrun --nproc_per_node=1 ./src/main.py --config_format base --model llama --distributed_backend nccl \
    --n_embd 768 --n_head 12 --n_layer 24 \
    --batch_size 64 --sequence_length 512 --acc_steps 4 \
    --dataset fineweb --iterations 128000 \
    --dropout 0.0 --warmup_steps 8000 --grad_clip 0.5 --seed 0 \
    --opt sf-adamw --lr 1e-3 --weight_decay 0.1 --scheduler none \
    --beta1 0.9 --beta2 0.9999 \
    --wandb --wandb_project YOUR_WANDB-PROJECT  --wandb_entity YOUR-WANDB-ENTITY \
    --eval_interval 115 --latest_ckpt_interval 1000 \