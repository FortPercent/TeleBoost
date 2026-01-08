#!/bin/bash
set -e

########################################
# 0. 参数解析（关键新增）
########################################

NODE_RANK=""
NNODES=""
MASTER_ADDR=""
MASTER_PORT="11322"

while [[ $# -gt 0 ]]; do
    case $1 in
        --node_rank)
            NODE_RANK="$2"
            shift 2
            ;;
        --nnodes)
            NNODES="$2"
            shift 2
            ;;
        --master_addr)
            MASTER_ADDR="$2"
            shift 2
            ;;
        --master_port)
            MASTER_PORT="$2"
            shift 2
            ;;
        *)
            break
            ;;
    esac
done

# 允许用环境变量兜底（方便 SLURM / Ray / SSH）
NODE_RANK=${NODE_RANK:-${RANK:-0}}
NNODES=${NNODES:-${NNODES_ENV:-""}}
MASTER_ADDR=${MASTER_ADDR:-${MASTER_ADDR_ENV:-"10.244.48.160"}}

if [[ -z "$NNODES" ]]; then
    echo "[ERROR] --nnodes is required"
    exit 1
fi

echo "[INFO] NODE_RANK=$NODE_RANK"
echo "[INFO] NNODES=$NNODES"
echo "[INFO] MASTER_ADDR=$MASTER_ADDR"
echo "[INFO] MASTER_PORT=$MASTER_PORT"

########################################
# 1. 环境变量（原样）
########################################

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export CUDA_VISIBLE_DEVICES=2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/Megatron-LM

########################################
# 2. 并行配置（原样）
########################################

CP=4
TP=1

N_MOE=1
# N_LAYERS=1
N_GPU_FOR_TRAIN=16
N_GPU_FOR_DATA=4

TENSORBOARD_LOGS_PATH=./logs
CHECKPOINT_PATH=/nvfile-heatstorage/yuc/refactor/Teletron/test
mkdir -p $CHECKPOINT_PATH

########################################
# 3. 推导参数（略微整理，逻辑不变）
########################################

MBS=1
N_GPU=$((N_GPU_FOR_TRAIN + N_GPU_FOR_DATA))
WORLD_SIZE=$N_GPU_FOR_TRAIN
N_VAE=$N_GPU_FOR_DATA
GBS=$(($WORLD_SIZE * $MBS / $CP / $TP))

TOTAL_MOE_NODES=$((NNODES - N_VAE / 8))
NODES_PER_MOE=$((TOTAL_MOE_NODES / N_MOE))
I_MOE=$((NODE_RANK / NODES_PER_MOE))

if [ "$NNODES" -eq 1 ]; then
    N_PROC=$N_GPU
else
    N_PROC=8
fi

########################################
# 4. MOE 参数（原样）
########################################

if [ $N_MOE -eq 1 ]; then
    MOE_ARGS=(--moe-step-factor-list 0.0 --moe-step-factor-list 1.0)
elif [ $N_MOE -eq 2 ]; then
    MOE_ARGS=(--moe-step-factor-list 0.0 --moe-step-factor-list 0.833 --moe-step-factor-list 1.0)
elif [ $N_MOE -eq 4 ]; then
    MOE_ARGS=(
        --moe-step-factor-list 0.0
        --moe-step-factor-list 0.625
        --moe-step-factor-list 0.833
        --moe-step-factor-list 0.937
        --moe-step-factor-list 1.0
    )
else
    echo "N_MOE must be 1, 2 or 4"
    exit 1
fi

########################################
# 5. torchrun 参数（关键）
########################################

DISTRIBUTED_ARGS=(
    --nproc_per_node $N_PROC
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

########################################
# 6. 其余参数（完全原样）
########################################

MERGE_FILE=/nvfile-heatstorage/teleai-infra/wxe/Megatron-LM/data/gpt_2_merge.txt
DATA_PATH=./checkpoint

GPT_MODEL_ARGS=( ... )
TRAINING_ARGS=( ... )
MODEL_PARALLEL_ARGS=( ... )
DATA_ARGS=( ... )
EVAL_AND_LOGGING_ARGS=(
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --save $CHECKPOINT_PATH/node_$I_MOE
)

########################################
# 7. 启动
########################################

torchrun "${DISTRIBUTED_ARGS[@]}" examples/teleai/pretrain_i2v.py \
    "${GPT_MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}" \
    "$@"
