#!/usr/bin/env bash
# =============================================================================
# 4×H800 80GB smoke test (1.3B + Qwen-VL reward variant)
#
# 基于 run_dancegrpo_1p3B_4gpu_smoke.sh, 把 reward_model 从 HPSv2 换成
# Qwen2.5-VL-7B-Instruct (而不是原 wxe qwen.sh 用的 32B, 4 卡装不下两个 32B):
#   - reward_model.type = "qwen"
#   - reward_model.model.path = /gfs/platform/public/infra/Qwen2.5-VL-7B-Instruct
# =============================================================================
set -xeuo pipefail

project_name='Dance-grpo'

export TIMESTAMP=$(date +"%m-%d_%H-%M-%S")
exp_name=${project_name}_SMOKE_4gpu_${TIMESTAMP}

adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 20))

loss_agg_mode="token-mean"

enable_filter_groups=True
filter_groups_metric=acc
max_num_gen_batches=10

# === SMOKE: 缩小 batch ===
train_prompt_bsz=2
gen_prompt_bsz=$((train_prompt_bsz * 3))
n_resp_per_prompt=2

# Ray
WORKING_DIR=${WORKING_DIR:-"${PWD}"}
RUNTIME_ENV=${RUNTIME_ENV:-"${WORKING_DIR}/verl/trainer/runtime_env.yaml"}
NNODES=${NNODES:-1}

# Paths (沿用 wxe 部署的 GFS 路径)
RAY_DATA_HOME="/gfs/platform/public/infra/wxe"
CKPTS_DIR=${CKPTS_DIR:-"/tmp/dancegrpo_smoke_ckpt"}
TRAIN_FILE=${TRAIN_FILE:-"/gfs/space/chatrl/users/wxe/fastvideo/data/processed_wan_prompt.json"}
TEST_FILE=${TEST_FILE:-"/gfs/space/chatrl/users/wxe/fastvideo/data/processed_wan_prompt.json"}

export TENSORBOARD_DIR=/tmp/dancegrpo_smoke_tb/${exp_name}

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# Performance Related Parameter
sp_size=1
use_dynamic_bsz=False
actor_ppo_max_token_len=$((max_prompt_length + max_response_length))
infer_ppo_max_token_len=$((max_prompt_length + max_response_length))
offload=True
gen_tp=1

HYDRA_FULL_ERROR=1 python3 -m recipe.dancegrpo.main_dancegrpo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.train_batch_size=${train_prompt_bsz} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path='/tmp/Wan2.1-T2V-1.3B' \
    +actor_rollout_ref.model.wan_version='wan21' \
    actor_rollout_ref.model.vae_model_path='/tmp/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth' \
    actor_rollout_ref.cfg=5.0 \
    actor_rollout_ref.h=256 \
    actor_rollout_ref.w=256 \
    actor_rollout_ref.num_frames=9 \
    actor_rollout_ref.sampling_steps=1 \
    actor_rollout_ref.actor.eta=0.25 \
    actor_rollout_ref.lr_warmup_steps=0 \
    actor_rollout_ref.use_hpsv2=True \
    actor_rollout_ref.shift=5 \
    actor_rollout_ref.actor.timestep_fraction=0.6 \
    actor_rollout_ref.init_same_noise=True \
    actor_rollout_ref.actor.clip_range=1e-4 \
    actor_rollout_ref.actor.adv_clip_max=5.0 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k="${top_k}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=-1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    reward_model.enable=True \
    reward_model.type="qwen" \
    reward_model.rollout.load_format=safetensors \
    reward_model.micro_batch_size_per_gpu=1 \
    reward_model.model.path='/gfs/platform/public/infra/Qwen2.5-VL-32B-Instruct' \
    reward_model.rollout.gpu_memory_utilization=0.6 \
    reward_model.model.input_tokenizer=null \
    reward_model.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    reward_model.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    trainer.logger="tensorboard" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.total_training_steps=2 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=False \
    trainer.test_freq=999 \
    trainer.save_freq=999 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.type="diffusion" \
    trainer.balance_batch=False
