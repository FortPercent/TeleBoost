#!/bin/bash
set -e

#######################################
# 0. 参数解析（新增，不影响原逻辑）
#######################################

NODE_RANK=""
NNODES=""
MASTER_ADDR="10.244.67.246"
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

# 环境变量兜底（兼容 SLURM / Ray / SSH）
NODE_RANK=${NODE_RANK:-${RANK:-0}}
NNODES=${NNODES:-${NNODES_ENV:-""}}
MASTER_ADDR=${MASTER_ADDR:-${MASTER_ADDR_ENV:-"127.0.0.1"}}

if [[ -z "$NNODES" ]]; then
    echo "[ERROR] --nnodes is required"
    exit 1
fi

echo "[INFO] MASTER_ADDR=$MASTER_ADDR"
echo "[INFO] MASTER_PORT=$MASTER_PORT"
echo "[INFO] NODE_RANK=$NODE_RANK"
echo "[INFO] NNODES=$NNODES"

#######################################
# 1. 环境变量（原样）
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

#######################################
# 2. 并行配置（原样）
#######################################

CP=4
TP=1   # not support

# Multi-node config
N_MOE=1
N_LAYERS=1
N_GPU_FOR_TRAIN=12
N_GPU_FOR_DATA=3

TENSORBOARD_LOGS_PATH=./logs
CHECKPOINT_PATH=/nvfile-heatstorage/yuc/refactor/Teletron/test
mkdir -p $CHECKPOINT_PATH

#######################################
# 3. 推导参数（保持你原逻辑）
#######################################

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
elif [ "$NODE_RANK" -eq $((NNODES - 1)) ]; then
    # 最后一个节点：VAE / Data
    N_PROC=$N_VAE
else
    # 训练 / MOE 节点
    N_PROC=8
fi


#######################################
# 4. MOE 参数（原样）
#######################################

if [ $N_MOE -eq 1 ]; then
    MOE_ARGS=(
        --moe-step-factor-list 0.0
        --moe-step-factor-list 1.0
    )
elif [ $N_MOE -eq 2 ]; then
    MOE_ARGS=(
        --moe-step-factor-list 0.0
        --moe-step-factor-list 0.833
        --moe-step-factor-list 1.0
    )
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

#######################################
# 5. 打印关键信息（保留）
#######################################

echo '$MASTER_ADDR' $MASTER_ADDR
echo '$I_MOE & $N_MOE' $I_MOE $N_MOE
echo '$NODE_RANK & $NNODES' $NODE_RANK $NNODES
echo '$N_GPU_FOR_TRAIN' $N_GPU_FOR_TRAIN
echo '$N_GPU_FOR_DATA' $N_GPU_FOR_DATA

#######################################
# 6. 数据 & 分布式参数
#######################################

MERGE_FILE=/nvfile-heatstorage/teleai-infra/wxe/Megatron-LM/data/gpt_2_merge.txt
DATA_PATH=./checkpoint

DISTRIBUTED_ARGS=(
    --nproc_per_node $N_PROC
    --nnodes $NNODES
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

#######################################
# 7. Megatron 参数（原样）
#######################################

GPT_MODEL_ARGS=(
    --num-layers $N_LAYERS
    --hidden-size 5120
    --num-attention-heads 40
    # --has-image-input
    --seq-length 512
    --max-position-embeddings 4096
    --tokenizer-type NullTokenizer
    --vocab-size 0
)

TRAINING_ARGS=(
    --micro-batch-size ${MBS}
    --train-iters 100000
    --weight-decay 1e-3
    --init-method-std 0.006
    --clip-grad 0.0
    --bf16
    --lr 1e-5
    --lr-decay-style cosine
    --lr-warmup-fraction 0
    --recompute-granularity full
    --recompute-method block
    --activation-offload
    # --use-distributed-optimizer
    --use-zero2
    --recompute-num-layers 40
    --no-rope-fusion
    --distributed-timeout-minutes 60
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --context-parallel-size ${CP}
    --distributed-vae
    --distributed-vae-world-size $N_VAE
    --consumer-models-num $N_MOE
    --producer-batch-size 1
)

DATA_ARGS=(
    # --dataset-type VastDataset
    --data-path $DATA_PATH
    --merge-file $MERGE_FILE
    --split 949,50,1
    --dataloader-type single
    --num-workers 1
)

EVAL_AND_LOGGING_ARGS=(
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
    --tensorboard-log-interval 1
    --tensorboard-queue-size 10
    --log-interval 1
    --save-interval 2000
    --eval-interval 2000
    --save $CHECKPOINT_PATH/node_$I_MOE
    --eval-iters 2
)

#######################################
# 8. 启动
#######################################

torchrun "${DISTRIBUTED_ARGS[@]}" examples/teleai/pretrain_i2v.py \
    "${GPT_MODEL_ARGS[@]}" \
    "${TRAINING_ARGS[@]}" \
    "${MODEL_PARALLEL_ARGS[@]}" \
    "${MOE_ARGS[@]}" \
    "${DATA_ARGS[@]}" \
    "${EVAL_AND_LOGGING_ARGS[@]}" \
    "$@"
