#!/bin/bash

cd /path/to/your/repo/llm-baselines
pip install -r requirements.txt

export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
python ./src/main.py --model base --n_embd 768 --n_head 12 --wandb --n_layer 24 --batch_size 50 --sequence_length 512 --wandb --acc_steps 1 --dropout 0.05

# for cpu
python ./src/main.py --n_layer=2 --n_head=4 --n_embd=128 --sequence_length=256 --dataset=shakespeare-char --device=cpu --vocab_size=96

# for quick gpu test
python ./src/main.py --n_layer=2 --n_head=4 --n_embd=128 --sequence_length=256 --dataset=shakespeare-char --vocab_size=96

# for full training with shakespeare
export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
python ./src/main.py --dataset=shakespeare-char --n_embd 768 --n_head 12 --wandb --n_layer 24 --batch_size 100 --sequence_length 512 --wandb --acc_steps 1 --dropout 0.05


# 2024-08-16
# dc
python ./src/main_dc.py --dataset=shakespeare-char --n_embd 768 --n_head 12 --n_layer 24 --batch_size 100 --sequence_length 512 --acc_steps 1 --dropout 0.05 --wandb
# bm
python ./src/main.py --dataset=shakespeare-char --n_embd 768 --n_head 12 --n_layer 24 --batch_size 100 --sequence_length 512 --acc_steps 1 --dropout 0.05 --wandb
# nz
python ./src/main_nz.py --dataset=shakespeare-char --n_embd 768 --n_head 12 --n_layer 24 --batch_size 100 --sequence_length 512 --acc_steps 1 --dropout 0.05 --wandb
# cr
python ./src/main_cr.py --dataset=shakespeare-char --n_embd 768 --n_head 12 --n_layer 24 --batch_size 100 --sequence_length 512 --acc_steps 1 --dropout 0.05 --wandb

# slimpajama
# dc
python ./src/main_dc.py --dataset=slimpajama --n_embd 768 --n_head 12 --n_layer 24 --batch_size 100 --sequence_length 512 --acc_steps 1 --dropout 0.05 --wandb
# bm
python ./src/main.py --dataset=slimpajama --n_embd 768 --n_head 12 --n_layer 24 --batch_size 100 --sequence_length 512 --acc_steps 1 --dropout 0.05 --wandb
# nz
python ./src/main_nz.py --dataset=slimpajama --n_embd 768 --n_head 12 --n_layer 24 --batch_size 100 --sequence_length 512 --acc_steps 1 --dropout 0.05 --wandbd