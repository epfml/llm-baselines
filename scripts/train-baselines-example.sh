#!/bin/bash

cd /path/to/your/repo/llm-baselines
pip install -r requirements.txt

export WANDB_API_KEY="put your authorize key here, to find it: https://wandb.ai/authorize"
python ./src/main.py --model base --n_embd 768 --n_head 12 --wandb_run_prefix h768_nh12_nlyr24_sl512_d005 --n_layer 24 --batch_size 50 --sequence_length 512 --wandb --acc_steps 4 --dropout 0.05

# for cpu
python ./src/main.py --n_layer=2 --n_head=4 --n_embd=128 --sequence_length=256 --dataset=shakespeare-char --device=cpu --vocab_size=96

# for quick gpu test
python ./src/main.py --n_layer=2 --n_head=4 --n_embd=128 --sequence_length=256 --dataset=shakespeare-char --vocab_size=96