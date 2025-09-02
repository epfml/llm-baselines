torchrun --nproc_per_node=1 ./src/main.py --config_format base --model llama --distributed_backend nccl \
    --n_embd 2048 --n_head 16 --n_layer 12 \
    --batch_size 62 --sequence_length 512 --acc_steps 32 \
    --dataset fineweb --iterations 48000 \
    --dropout 0.0 --warmup_steps 2000 --grad_clip 0.1 --seed 0 \
    --opt muon --lr 1e-3 --muon_lr_factor 1e-2 --weight_decay 0.1 --scheduler cos \
    --beta1 0.8 --beta2 0.999 --momentum 0.95 --nesterov True --muon_ns_steps 5 \
    --wandb --wandb_project YOUR_WANDB-PROJECT  --wandb_entity YOUR-WANDB-ENTITY \
    --eval_interval 200 --permanent_ckpt_interval 2000 \