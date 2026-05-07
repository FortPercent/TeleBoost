#!/usr/bin/env bash
# Wan DPO training launcher.
#
# Usage (single node, 8 GPUs):
#   export MEGATRON_LM_DIR=/path/to/Megatron-LM
#   export TELEAI_DATA_TOOL_DIR=/path/to/teleai_data_tool
#   export EXPR_NAME=my_run
#   bash examples/teleai/train_dpo.sh
#
# Usage (multi-node):
#   Set NNODES, NODE_RANK (or RANK), MASTER_ADDR additionally.
#
# All paths/parallelism config below can be overridden via env vars.

set -euo pipefail

# ─── Required: external repos on PYTHONPATH ────────────────────────────
: "${MEGATRON_LM_DIR:?set MEGATRON_LM_DIR to a Megatron-LM checkout}"
: "${TELEAI_DATA_TOOL_DIR:?set TELEAI_DATA_TOOL_DIR to a teleai_data_tool checkout}"
export PYTHONPATH="${PYTHONPATH:-}:${MEGATRON_LM_DIR}:${TELEAI_DATA_TOOL_DIR}"
[ -n "${VIDEO_DEPTH_ANYTHING_DIR:-}" ] && export PYTHONPATH="${PYTHONPATH}:${VIDEO_DEPTH_ANYTHING_DIR}"

# ─── Experiment name & checkpoint dirs ─────────────────────────────────
EXPR_NAME="${EXPR_NAME:-wan_dpo_default}"
CHECKPOINT_PATH_SAVE="${CHECKPOINT_PATH_SAVE:-./checkpoints/${EXPR_NAME}}"
CHECKPOINT_PATH_LOAD="${CHECKPOINT_PATH_LOAD:-${CHECKPOINT_PATH_SAVE}}"
TENSORBOARD_LOGS_PATH="${TENSORBOARD_LOGS_PATH:-./logs/${EXPR_NAME}}"
mkdir -p "${CHECKPOINT_PATH_SAVE}" "${TENSORBOARD_LOGS_PATH}"

# ─── Distributed config ────────────────────────────────────────────────
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-11322}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-${RANK:-0}}"
N_PROC="${N_PROC:-8}"

# ─── Parallelism (CP × TP × distributed-VAE × MoE-consumers) ──────────
CP="${CP:-8}"
TP="${TP:-1}"            # currently must be 1 (TP not supported)
N_VAE="${N_VAE:-2}"      # ranks dedicated to VAE encoder (consumer)
N_MOE="${N_MOE:-1}"      # consumer model copies (1 = no MoE)

# ─── Runtime envs ──────────────────────────────────────────────────────
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ─── Entry point and config (override via positional args) ─────────────
TRAIN_SCRIPT="${1:-examples/teleai/pretrain_dpo_i2v.py}"
CONFIG_PATH="${2:-config.wan_dpo.config}"
[ "$#" -ge 1 ] && shift
[ "$#" -ge 1 ] && shift

echo "Launching: ${TRAIN_SCRIPT}"
echo "  config:        ${CONFIG_PATH}"
echo "  expr_name:     ${EXPR_NAME}"
echo "  CP / TP:       ${CP} / ${TP}"
echo "  N_VAE / N_MOE: ${N_VAE} / ${N_MOE}"
echo "  ${NNODES} node(s) × ${N_PROC} proc"

torchrun \
    --nproc_per_node "${N_PROC}" \
    --nnodes "${NNODES}" \
    --node_rank "${NODE_RANK}" \
    --master_addr "${MASTER_ADDR}" \
    --master_port "${MASTER_PORT}" \
    "${TRAIN_SCRIPT}" \
    --micro-batch-size 1 \
    --train-iters 200000 \
    --weight-decay 1e-4 \
    --init-method-std 0.006 \
    --clip-grad 1.0 \
    --bf16 \
    --lr 1e-5 \
    --lr-decay-style constant \
    --lr-warmup-fraction 0 \
    --recompute-granularity full \
    --recompute-method block \
    --recompute-num-layers 40 \
    --use-zero2 \
    --no-rope-fusion \
    --distributed-timeout-minutes 60 \
    --override-opt_param-scheduler \
    --data-parallel-random-init \
    --tensor-model-parallel-size "${TP}" \
    --context-parallel-size "${CP}" \
    --distributed-vae \
    --distributed-vae-world-size "${N_VAE}" \
    --consumer-models-num "${N_MOE}" \
    --split 949,50,1 \
    --num-workers 2 \
    --config-path "${CONFIG_PATH}" \
    --tensorboard-dir "${TENSORBOARD_LOGS_PATH}" \
    --tensorboard-log-interval 1 \
    --tensorboard-queue-size 10 \
    --log-interval 1 \
    --save-interval 500 \
    --eval-iters 0 \
    `# DPO eval is unsupported (forward_step returns a 5-element list that` \
    `# megatron's eval reducer can't divide). pretrain_dpo_i2v.py asserts` \
    `# this at startup; setting eval-iters=0 keeps the launcher quiet.` \
    --producer-log-level 1 \
    "$@"
