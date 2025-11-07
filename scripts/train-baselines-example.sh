#!/bin/bash

cd /scratch/homes/sfan/llm-baselines
pip install -r requirements.txt

export WANDB_API_KEY="put your wandb api key here"
python ./src/main.py --model base --n_embd 768 --n_head 6 --wandb_run_prefix h768_nh12_nlyr24_sl512_d005 --n_layer 6 --batch_size 50 --sequence_length 512 --wandb --acc_steps 4 --dropout 0.05
