#!/usr/bin/env bash
set -euo pipefail

# Public launcher for TeleBoost.
#
# Required:
#   TRAIN_FILE=/path/to/processed_wan_prompt.json
#   TEST_FILE=/path/to/processed_wan_prompt.json
#   WAN_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B
#   REWARD_MODEL_PATH=/path/to/HPS_v2.1_compressed.pt
#
# Optional:
#   TELEBOOST_METHOD=default|bgpo|vipo|joint
#   WAN_VERSION=wan21|wan22
#   WAN_VAE_PATH=/path/to/Wan2.1_VAE.pth
#
# Orthogonal flags (any combination, on top of TELEBOOST_METHOD):
#   SP_SIZE=1|2|4|8                          (Wan Ulysses sequence parallel)
#   INIT_SAME_NOISE=True|False               (False = real reward variance for grad checks)
#   ENABLE_GRPOGUARD=True|False              (turns on GRPO-Guard ratio_norm + grad_reweight)
#   ENABLE_FLOWGRPO=True|False               (turns on flow-grpo SDE; also bumps SAMPLING_STEPS to >=4)
#   ADV_ESTIMATOR=grpo                       (algorithm.adv_estimator; only "grpo" is wired up)
#   SAMPLING_STEPS=N                         (rollout denoise steps; HPS needs >=4 to avoid nan)
#   TOTAL_TRAINING_STEPS=N
#   N_GPUS_PER_NODE=N
#   N_RESP_PER_PROMPT=N  TRAIN_PROMPT_BSZ=N  PPO_MINI_BATCH_SIZE=N
#   VIDEO_HEIGHT=N  VIDEO_WIDTH=N  NUM_FRAMES=N
#
# Multi-node (run the same command on every node; the script self-routes
# based on NODE_RANK):
#   NNODES=N                                 (total number of nodes; default 1)
#   NODE_RANK=0..NNODES-1                    (this node's rank; 0 = master)
#   MASTER_ADDR=hostname-or-ip               (rank-0 host, reachable from workers)
#   MASTER_PORT=6379                         (Ray head port; default 6379)

: "${TRAIN_FILE:?Set TRAIN_FILE=/path/to/processed_wan_prompt.json}"
: "${TEST_FILE:?Set TEST_FILE=/path/to/processed_wan_prompt.json}"
: "${WAN_MODEL_PATH:?Set WAN_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B}"

method="${TELEBOOST_METHOD:-default}"
case "${method}" in
  default|bgpo|vipo|joint) ;;
  *)
    echo "Unsupported TELEBOOST_METHOD='${method}'. Use default, bgpo, vipo, or joint." >&2
    exit 2
    ;;
esac

project_name="${PROJECT_NAME:-TeleBoost-DanceGRPO}"
timestamp="$(date +"%m-%d_%H-%M-%S")"
exp_name="${EXPERIMENT_NAME:-${project_name}_${method}_${timestamp}}"

working_dir="${WORKING_DIR:-"${PWD}"}"
output_dir="${TELEBOOST_OUTPUT_DIR:-"${working_dir}/outputs"}"
ckpts_dir="${CKPTS_DIR:-"${output_dir}/checkpoints/${exp_name}"}"
export TENSORBOARD_DIR="${TENSORBOARD_DIR:-"${output_dir}/tensorboard/${exp_name}"}"

wan_version="${WAN_VERSION:-wan22}"
wan_vae_path="${WAN_VAE_PATH:-"${WAN_MODEL_PATH}/Wan2.1_VAE.pth"}"
reward_model_name="${REWARD_MODEL_NAME:-hps}"
reward_model_path="${REWARD_MODEL_PATH:-}"

nnodes="${NNODES:-1}"
n_gpus="${N_GPUS_PER_NODE:-8}"
train_prompt_bsz="${TRAIN_PROMPT_BSZ:-8}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-3}"
total_training_steps="${TOTAL_TRAINING_STEPS:-1000}"

height="${VIDEO_HEIGHT:-480}"
width="${VIDEO_WIDTH:-832}"
num_frames="${NUM_FRAMES:-49}"
# sampling_steps=16 matches the upstream DanceGRPO recipe.  The rollout
# drops the final sigma->0 step (numerically unstable for log-prob), so
# train_timesteps = max(1, int((sampling_steps-1) * timestep_fraction));
# values <=2 silently no-op the policy update for the default
# fraction=0.6, and <4 trips HPS into nan on near-noise inputs.
sampling_steps="${SAMPLING_STEPS:-16}"

max_prompt_length="${MAX_PROMPT_LENGTH:-2048}"
max_response_length="${MAX_RESPONSE_LENGTH:-20480}"
ppo_token_len=$((max_prompt_length + max_response_length))

enable_bgpo=False
use_rerange=False
pixel_weight_enable=False
reward_type=single

if [[ "${method}" == "bgpo" ]]; then
  enable_bgpo=True
  use_rerange=True
fi

if [[ "${method}" == "vipo" ]]; then
  pixel_weight_enable=True
fi

# Orthogonal flags (independent of TELEBOOST_METHOD).
sp_size="${SP_SIZE:-1}"
init_same_noise="${INIT_SAME_NOISE:-True}"
enable_grpoguard="${ENABLE_GRPOGUARD:-False}"
enable_flowgrpo="${ENABLE_FLOWGRPO:-False}"
adv_estimator="${ADV_ESTIMATOR:-grpo}"

# flow-grpo's SDE solver needs more than one denoising step; auto-bump if user
# left SAMPLING_STEPS at the default of 1.
if [[ "${enable_flowgrpo}" == "True" && "${sampling_steps}" -le 1 ]]; then
  sampling_steps=4
fi

if [[ "${method}" == "joint" ]]; then
  reward_type=joint
  : "${JOINT_AESTHETIC_CLIP_PATH:?Set JOINT_AESTHETIC_CLIP_PATH for joint reward runs}"
  : "${JOINT_AESTHETIC_MODEL_PATH:?Set JOINT_AESTHETIC_MODEL_PATH for joint reward runs}"
  : "${JOINT_RAFT_MODEL_PATH:?Set JOINT_RAFT_MODEL_PATH for joint reward runs}"
  : "${JOINT_VIDEOCLIP_MODEL_PATH:?Set JOINT_VIDEOCLIP_MODEL_PATH for joint reward runs}"
  : "${JOINT_VIDEOPHY_MODEL_PATH:?Set JOINT_VIDEOPHY_MODEL_PATH for joint reward runs}"
else
  : "${reward_model_path:?Set REWARD_MODEL_PATH=/path/to/reward_model.pt for non-joint runs}"
fi

overrides=(
  "data.train_files=${TRAIN_FILE}"
  "data.val_files=${TEST_FILE}"
  "data.prompt_key=prompt"
  "data.truncation=left"
  "data.train_batch_size=${train_prompt_bsz}"
  "data.max_prompt_length=${max_prompt_length}"
  "data.max_response_length=${max_response_length}"
  "algorithm.adv_estimator=${adv_estimator}"
  "algorithm.use_kl_in_reward=False"
  "algorithm.kl_ctrl.kl_coef=0.0"
  "algorithm.bgpo.enable=${enable_bgpo}"
  "algorithm.bgpo.use_rerange=${use_rerange}"
  "algorithm.bgpo.append_rerange_samples=False"
  # CRT (paper Eq. 4): R̃ = [λ·(R − R_prior) + 𝟙{R > R_prior}] · exp(R)
  "algorithm.bgpo.lambda_contrast=${BGPO_LAMBDA_CONTRAST:-1.0}"
  # RAS (paper Eq. 2): w = 1 + α·[2σ(k·(R̄ − R_prior)) − 1]
  "algorithm.bgpo.adaptive_weight_method=${BGPO_ADAPTIVE_WEIGHT_METHOD:-paper}"
  "algorithm.bgpo.k_sharpness=${BGPO_K_SHARPNESS:-1.0}"
  "algorithm.bgpo.regularization_term_alpha=${BGPO_ALPHA:-0.5}"
  "algorithm.bgpo.min_adv_scale=${BGPO_MIN_ADV_SCALE:-0.01}"
  "algorithm.bgpo.max_adv_scale=${BGPO_MAX_ADV_SCALE:-10.0}"
  "algorithm.bgpo.exp_clamp=${BGPO_EXP_CLAMP:-30.0}"
  "actor_rollout_ref.model.path=${WAN_MODEL_PATH}"
  "+actor_rollout_ref.model.wan_version=${wan_version}"
  "actor_rollout_ref.model.vae_model_path=${wan_vae_path}"
  "actor_rollout_ref.cfg=${GUIDANCE_SCALE:-5.0}"
  "actor_rollout_ref.h=${height}"
  "actor_rollout_ref.w=${width}"
  "actor_rollout_ref.num_frames=${num_frames}"
  "actor_rollout_ref.sampling_steps=${sampling_steps}"
  "actor_rollout_ref.actor.eta=${ACTOR_ETA:-0.25}"
  "actor_rollout_ref.lr_warmup_steps=0"
  "actor_rollout_ref.use_hpsv2=True"
  "actor_rollout_ref.shift=${SHIFT:-5}"
  "actor_rollout_ref.actor.timestep_fraction=${TIMESTEP_FRACTION:-0.6}"
  "actor_rollout_ref.init_same_noise=${init_same_noise}"
  "actor_rollout_ref.actor.grpo_guard.enable=${enable_grpoguard}"
  "actor_rollout_ref.actor.grpo_guard.ratio_norm=${enable_grpoguard}"
  "actor_rollout_ref.actor.grpo_guard.grad_reweight=${enable_grpoguard}"
  "actor_rollout_ref.flow_grpo.enable=${enable_flowgrpo}"
  "actor_rollout_ref.flow_grpo.sde_window_size=${FLOWGRPO_SDE_WINDOW_SIZE:-2}"
  "actor_rollout_ref.actor.clip_range=1e-4"
  "actor_rollout_ref.actor.adv_clip_max=5.0"
  "actor_rollout_ref.actor.use_kl_loss=False"
  "actor_rollout_ref.actor.kl_loss_coef=0.0"
  "actor_rollout_ref.actor.clip_ratio_low=0.2"
  "actor_rollout_ref.actor.clip_ratio_high=0.28"
  "actor_rollout_ref.actor.clip_ratio_c=10.0"
  "actor_rollout_ref.model.use_remove_padding=True"
  "actor_rollout_ref.model.enable_gradient_checkpointing=True"
  "actor_rollout_ref.actor.use_dynamic_bsz=False"
  "actor_rollout_ref.ref.log_prob_use_dynamic_bsz=False"
  "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=False"
  "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${ppo_token_len}"
  "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${ppo_token_len}"
  "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${ppo_token_len}"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2"
  "actor_rollout_ref.rollout.n=${n_resp_per_prompt}"
  "actor_rollout_ref.pixel_weight.enable=${pixel_weight_enable}"
  "actor_rollout_ref.pixel_weight.model_path=${PIXEL_WEIGHT_MODEL_PATH:-facebook/dinov2-large}"
  "actor_rollout_ref.pixel_weight.pca_method=${PIXEL_WEIGHT_PCA_METHOD:-weighted}"
  "actor_rollout_ref.pixel_weight.sigma=${PIXEL_WEIGHT_SIGMA:-1.0}"
  "actor_rollout_ref.pixel_weight.kl_loss_compatible=False"
  "actor_rollout_ref.actor.optim.lr=${ACTOR_LR:-2e-6}"
  "actor_rollout_ref.actor.optim.lr_warmup_steps=0"
  "actor_rollout_ref.actor.optim.weight_decay=0.1"
  # Default to ``n_gpus`` so verl's world-size normalization
  # (``ppo_mini_batch_size //= world_size``) doesn't floor to 0.
  # Override via ``PPO_MINI_BATCH_SIZE`` for non-default sizing.
  "actor_rollout_ref.actor.ppo_mini_batch_size=${PPO_MINI_BATCH_SIZE:-${n_gpus}}"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.fsdp_config.param_offload=${FSDP_OFFLOAD:-True}"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=${FSDP_OFFLOAD:-True}"
  "actor_rollout_ref.actor.fsdp_config.fsdp_size=-1"
  "actor_rollout_ref.actor.entropy_coeff=0"
  "actor_rollout_ref.actor.grad_clip=1.0"
  "actor_rollout_ref.actor.loss_agg_mode=token-mean"
  "actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size}"
  "actor_rollout_ref.ref.fsdp_config.param_offload=${FSDP_OFFLOAD:-True}"
  "actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size}"
  "actor_rollout_ref.rollout.ulysses_sequence_parallel_size=${sp_size}"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.7}"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=${ROLLOUT_TP_SIZE:-1}"
  "actor_rollout_ref.rollout.enable_chunked_prefill=True"
  "actor_rollout_ref.rollout.max_num_batched_tokens=${ppo_token_len}"
  "actor_rollout_ref.rollout.temperature=${TEMPERATURE:-1.0}"
  "actor_rollout_ref.rollout.top_p=${TOP_P:-1.0}"
  "actor_rollout_ref.rollout.top_k=${TOP_K:--1}"
  "actor_rollout_ref.rollout.val_kwargs.temperature=${TEMPERATURE:-1.0}"
  "actor_rollout_ref.rollout.val_kwargs.top_p=${VAL_TOP_P:-0.7}"
  "actor_rollout_ref.rollout.val_kwargs.top_k=${TOP_K:--1}"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.rollout.val_kwargs.n=1"
  "reward_model.enable=True"
  "reward_model.type=${reward_type}"
  "reward_model.micro_batch_size_per_gpu=1"
  "trainer.logger=${TRAINER_LOGGER:-tensorboard}"
  "trainer.project_name=${project_name}"
  "trainer.experiment_name=${exp_name}"
  "trainer.total_training_steps=${total_training_steps}"
  "trainer.n_gpus_per_node=${n_gpus}"
  "trainer.nnodes=${nnodes}"
  "trainer.val_before_train=${VAL_BEFORE_TRAIN:-False}"
  "trainer.test_freq=999"
  "trainer.save_freq=999"
  "trainer.total_epochs=1"
  "trainer.default_local_dir=${ckpts_dir}"
  "trainer.resume_mode=auto"
  "trainer.type=diffusion"
  "trainer.balance_batch=False"
)

if [[ "${reward_type}" == "joint" ]]; then
  overrides+=(
    "reward_model.joint.models.aesthetic.extra_config.clip_model_path=${JOINT_AESTHETIC_CLIP_PATH}"
    "reward_model.joint.models.aesthetic.extra_config.aes_model_path=${JOINT_AESTHETIC_MODEL_PATH}"
    "reward_model.joint.models.raft.model_path=${JOINT_RAFT_MODEL_PATH}"
    "reward_model.joint.models.videoclip.model_path=${JOINT_VIDEOCLIP_MODEL_PATH}"
    "reward_model.joint.models.videophy.model_path=${JOINT_VIDEOPHY_MODEL_PATH}"
  )
else
  overrides+=(
    "reward_model.model_name=${reward_model_name}"
    "reward_model.model.path=${reward_model_path}"
    "reward_model.extra_config.model_type=${HPS_MODEL_TYPE:-ViT-H-14}"
  )
fi

# Multi-node: dispatch master vs worker before launching training.
# Same script runs on every node — NODE_RANK selects the role.
#   - master (rank 0): start Ray head, then fall through to main_teleboost
#   - worker (rank >0): join the Ray cluster and block; main_teleboost
#                       runs only on master, but Ray schedules workers
#                       across the joined nodes.
node_rank="${NODE_RANK:-0}"
master_addr="${MASTER_ADDR:-127.0.0.1}"
master_port="${MASTER_PORT:-6379}"

if [[ "${nnodes}" -gt 1 ]]; then
  if [[ "${node_rank}" == "0" ]]; then
    echo "[teleboost] master (rank 0): starting Ray head on ${master_addr}:${master_port}"
    ray start --head --port="${master_port}" --num-gpus="${n_gpus}"
  else
    echo "[teleboost] worker (rank ${node_rank}): joining Ray cluster at ${master_addr}:${master_port}"
    exec ray start --address="${master_addr}:${master_port}" --num-gpus="${n_gpus}" --block
  fi
fi

# Forward extra positional args ("$@") to Hydra so callers can append
# overrides like ``actor_rollout_ref.actor.sigma_form=flow_grpo`` or
# ``data.prompt_key=caption`` without editing this script.
HYDRA_FULL_ERROR=1 python3 -m recipe.teleboost.main_teleboost "${overrides[@]}" "$@"
