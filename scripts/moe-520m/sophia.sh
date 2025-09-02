torchrun --nproc_per_node=1 ./src/main.py --config_format base --model llama --distributed_backend nccl \
    --n_embd 768 --n_head 12 --n_layer 12 --moe \
    --batch_size 32 --sequence_length 512 --acc_steps 8 \
    --dataset fineweb --iterations 336000 \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.5 --seed 0 \
    --opt sophiag --lr 5e-4 --weight_decay 0.1 --scheduler cos \
    --beta1 0.9 --beta2 0.999 \
    --wandb --wandb_project YOUR_WANDB-PROJECT  --wandb_entity YOUR-WANDB-ENTITY \
    --eval_interval 115 --latest_ckpt_interval 1000 \