#!/bin/bash
set -e

#######################################
# 0. 参数解析（新增，其余不动）
#######################################

NODE_RANK=""
NNODES=""
MASTER_ADDR="10.244.48.175"
MASTER_PORT="11325"

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

# 环境变量兜底（兼容 SLURM / Ray）
NODE_RANK=${NODE_RANK:-${RANK:-0}}
NNODES=${NNODES:-${NNODES_ENV:-""}}
MASTER_ADDR=${MASTER_ADDR:-${MASTER_ADDR_ENV:-"10.244.21.39"}}

if [[ -z "$NNODES" ]]; then
    echo "[ERROR] --nnodes is required"
    exit 1
fi

echo "[INFO] MASTER_ADDR=$MASTER_ADDR"
echo "[INFO] MASTER_PORT=$MASTER_PORT"
echo "[INFO] NODE_RANK=$NODE_RANK"
echo "[INFO] NNODES=$NNODES"

#######################################
# Run model（原样）
#######################################

export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/Megatron-LM
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/yuc/env/teleai_data_tool
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/qiuyang/Video-Depth-Anything/
# export MEMORY_SNAPSHOT=1
# export PROF_SAVE_PATH="./dpo_train_profile"
####################################### IMPORTANT ARGS #######################################
# Parallel config
CP=2
TP=1 # not support

# Multi-node config
N_MOE=1
N_GPU_FOR_TRAIN=16
N_GPU_FOR_DATA=8

# EXPR_NAME=sr_720p
# EXPR_NAME=wwan_22_14b_720p_81_dpo_lr_1_5e_6_clipgrad_1
EXPR_NAME=test_dpo_i2v
# EXPR_NAME=expr_480p_bf16

TRAIN_SCRIPT=${1:-"examples/teleai/pretrain_dpo_i2v.py"}
CONFIG_PATH=${2:-"config.wan_dpo.config"}
if [ $# -gt 0 ]; then
    shift
fi
echo "Launching: $TRAIN_SCRIPT"

TENSORBOARD_LOGS_PATH=./logs/${EXPR_NAME}
# CHECKPOINT_PATH_LOAD=/nvfile-heatstorage/AIGC_H100/basemodel_exp/ckpts/ljq/init/high_noise_I2V
CHECKPOINT_PATH_LOAD=/workspace/high_noise_I2V
# CHECKPOINT_PATH_LOAD=/nvfile-heatstorage/ai_infra/code/fanyk1/yp/Teletron-dpo/wan_22_14b_720p_81_dpo_lr_2e_6_clipgrad_20
CHECKPOINT_PATH_SAVE=/nvfile-heatstorage/ai_infra/code/fanyk1/yp/Teletron-dpo/${EXPR_NAME}
####################################### IMPORTANT ARGS END #######################################

mkdir -p $CHECKPOINT_PATH_SAVE

#######################################
# 分布式推导（原逻辑，一行不改）
#######################################

MBS=1
N_GPU=$((N_GPU_FOR_TRAIN + N_GPU_FOR_DATA))
WORLD_SIZE=$N_GPU_FOR_TRAIN
N_VAE=$N_GPU_FOR_DATA
GBS=$(($WORLD_SIZE * $MBS / $CP / $TP))

GPUS_PER_NODE=8

# 训练占满的整节点数
FULL_TRAIN_NODES=$((N_GPU_FOR_TRAIN / GPUS_PER_NODE))

# 训练在下一个节点剩余的 GPU 数
REMAIN_TRAIN_GPU=$((N_GPU_FOR_TRAIN % GPUS_PER_NODE))

if [ "$NNODES" -eq 1 ]; then
    # 单节点：训练 + VAE 全上
    N_PROC=$((N_GPU_FOR_TRAIN + N_GPU_FOR_DATA))

elif [ "$NODE_RANK" -lt "$FULL_TRAIN_NODES" ]; then
    # 完全被训练占满的节点
    N_PROC=$GPUS_PER_NODE

elif [ "$NODE_RANK" -eq "$FULL_TRAIN_NODES" ]; then
    # 训练剩余 + VAE 共存节点
    N_PROC=$((REMAIN_TRAIN_GPU + N_GPU_FOR_DATA))

else
    # 后续节点不用启动进程
    N_PROC=0
fi


#######################################
# Debug 输出（原样）
#######################################

echo '$MASTER_ADDR' $MASTER_ADDR
echo '$NODE_RANK & $NNODES' $NODE_RANK $NNODES
echo '$N_GPU_FOR_TRAIN' $N_GPU_FOR_TRAIN
echo '$N_GPU_FOR_DATA' $N_GPU_FOR_DATA
echo '$N_PROC' $N_PROC

#######################################
# torchrun 参数
#######################################

DISTRIBUTED_ARGS=(
    --nproc_per_node $N_PROC
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

#######################################
# 训练参数（原样）
#######################################

TRAINING_ARGS=(
    --micro-batch-size ${MBS}
    --train-iters 200000
    --weight-decay 1e-4
    --init-method-std 0.006
    --clip-grad 1.0
    --bf16
    --lr 1.5e-6
    --lr-decay-style constant
    --lr-warmup-fraction 0
    --recompute-granularity full
    --recompute-method block
    # --activation-offload
    # --use-distributed-optimizer
    --use-zero2
    --recompute-num-layers 40
    --no-rope-fusion
    --distributed-timeout-minutes 60
    --override-opt_param-scheduler
    --data-parallel-random-init
    --save-inputs
    --save-inputs-interval 1
    --save-inputs-dir ./test_data/saved_inputs_${EXPR_NAME}
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --context-parallel-size ${CP}
    --distributed-vae
    --distributed-vae-world-size $N_VAE
    --consumer-models-num $N_MOE
)

DATA_ARGS=(
    --split 949,50,1
    --num-workers 2
    --config-path ${CONFIG_PATH}
)

EVAL_AND_LOGGING_ARGS=(
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --tensorboard-log-interval 1
    --tensorboard-queue-size 10
    --log-interval 1
    --save-interval 100
    --eval-interval 500
    --load $CHECKPOINT_PATH_LOAD
    # --pretrained_checkpoint $CHECKPOINT_PATH_LOAD
    --save $CHECKPOINT_PATH_SAVE
    --eval-iters 20
    --producer-log-level 1
)

#######################################
# 启动
#######################################

torchrun "${DISTRIBUTED_ARGS[@]}" ${TRAIN_SCRIPT} \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}" \
    "${LORA_CFG[@]}" \
    "$@"
