#!/bin/bash

# 创建Reward目录
mkdir -p /gemini/space/ljm/Wan2.1-main_72btest_for_more/Reward

# # 1. 先在后台启动cuda7上的Qwen奖励计算服务
# echo "🚀 Starting Qwen reward service on cuda:7..."
# CUDA_VISIBLE_DEVICES=6,7  /gemini/platform/public/zqni/miniconda3/envs/wan_diffusers_grpo/bin/python \
#     /gemini/space/ljm/Wan2.1-main_72btest_for_more/start_qwen_reward_service.py \
#     > /gemini/space/ljm/Wan2.1-main_72btest_for_more/Reward/qwen_service.log 2>&1 &
# QWEN_PID=$!
# echo "Qwen service PID: $QWEN_PID"

# # 等待5秒确保服务启动
# sleep 5
# echo "✓ Qwen reward service started, check log: Reward/qwen_service.log"

# 2. 使用cuda0-6启动GRPO训练
echo "🎯 Starting GRPO training on cuda:0-7..."
export NCCL_DEBUG=INFO
# export NCCL_IB_DISABLE=1
# export NCCL_P2P_DISABLE=1
# export NCCL_SHM_DISABLE=0
# export NCCL_BLOCKING_WAIT=1
# export NCCL_TIMEOUT=1800
# export CUDA_LAUNCH_BLOCKING=1

# 只使用前张卡（cuda:0-7）进行训练
torchrun --nproc_per_node=8 \
    /gemini/space/ljm/Wan2.1-main_72btest_for_more/train_grpo_wan_fsdp_new.py \
    --train_gpus="0,1,2,3,4,5,6,7" \
    --seed 42 \
    --pretrained_model_name_or_path /gemini/space/Wan2___1-T2V-1___3B \
    --cache_dir ./cache_dir \
    --data_json_path /gemini/space/wuxuaner/Dancegrpo/data/1__3B/rl_embeddings/processed_wan_prompt.json \
    --train_batch_size 1 \
    --dataloader_num_workers 4 \
    --gradient_accumulation_steps 4 \
    --max_train_steps 10000 \
    --learning_rate 1e-5 \
    --output_dir /gemini/space/ljm/Wan2.1-main_72btest_for_more/outputs\
    --h 480 \
    --w 832 \
    --t 81 \
    --sampling_steps 16 \
    --eta 0.5 \
    --lr_warmup_steps 0 \
    --sampler_seed 1223627 \
    --max_grad_norm 1.0 \
    --weight_decay 0.0001 \
    --num_generations 2 \
    --shift 5.0 \
    --timestep_fraction 0.6 \
    --init_same_noise \
    --clip_range 1e-4 \
    --adv_clip_max 5.0 \
    --use_group \
    --enable_gradient_checkpointing \
    --use_sequential_cfg \
    --use_bf16 \
    --use_file_based_reward \
    --qwen_reward_model_path /gemini/space/Qwen/Qwen2___5-VL-72B-Instruct \
    #--cpu_offload \