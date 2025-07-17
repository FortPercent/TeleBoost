#!/bin/bash

# Run model
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/teleai-infra/litian/Megatron-LM

####################################### 
# Parallel config 
CP=1
TP=1 # not support

# Multi-node config 
N_MOE=1
N_GPU_FOR_TRAIN=24
N_GPU_FOR_DATA=24

# Single-node config 
# N_MOE=1
# N_GPU_FOR_TRAIN=1
# N_GPU_FOR_DATA=1

# EXPR_NAME=sr_720p
EXPR_NAME=expr_480p_bf16

# MODEL_ARGS=(
#     --num-layers 30
#     --hidden-size 5120
#     --ffn-hidden-size 13824
#     --num-attention-heads 40
# ) # 10B I2V

MODEL_ARGS=(
    --num-layers 30
    --hidden-size 1536
    --ffn-hidden-size 8960
    --num-attention-heads 12
) # 1.3B I2V

TASK=teleai_i2v

TENSORBOARD_LOGS_PATH=./logs/${EXPR_NAME}
CHECKPOINT_PATH_LOAD=/nvfile-heatstorage/myk/Teletron/checkpoint/${EXPR_NAME}
CHECKPOINT_PATH_SAVE=/nvfile-heatstorage/myk/Teletron/checkpoint/${EXPR_NAME}
mkdir -p $CHECKPOINT_PATH_SAVE


####################################### 

MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
MASTER_PORT='11322'
NODE_RANK=${RANK:-'0'}

MBS=1
N_GPU=$((N_GPU_FOR_TRAIN+N_GPU_FOR_DATA))
NNODES=$((($N_GPU-1)/8+1))
WORLD_SIZE=$N_GPU_FOR_TRAIN
N_VAE=$N_GPU_FOR_DATA
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))

if [ $NNODES -eq 1 ]; then
    N_PROC=$N_GPU
else
    N_PROC=8
fi

echo '$MASTER_ADDR' $MASTER_ADDR
echo '$NODE_RANK & $NNODES' $NODE_RANK $NNODES
echo '$N_GPU_FOR_TRAIN' $N_GPU_FOR_TRAIN
echo '$N_GPU_FOR_DATA' $N_GPU_FOR_DATA

DISTRIBUTED_ARGS=(
    --nproc_per_node $N_PROC 
    --nnodes $NNODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)


TRAINING_ARGS=(
    --model ParallelTeleaiModel 
    --task-type ${TASK}
    --micro-batch-size ${MBS}
    --train-iters 200000
    --weight-decay 1e-4
    --init-method-std 0.006 
    --clip-grad 1.0
    --bf16
    --lr 2e-6
    --lr-decay-style constant
    --lr-warmup-fraction 0
    --recompute-granularity full 
    --recompute-method block 
    --activation-offload
    --use-distributed-optimizer
    --recompute-num-layers 40
    --no-rope-fusion
    --distributed-timeout-minutes 60
    --override-opt_param-scheduler
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --context-parallel-size ${CP}
    --distributed-vae
    --distributed-vae-world-size $N_VAE
    --consumer-models-num $N_MOE
    --temp-accelerate
)
DATA_ARGS=(
    --dataset-type VastDataset
    --split 949,50,1
    --dataloader-type single
    --num-workers 1
)

EVAL_AND_LOGGING_ARGS=(
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    --tensorboard-log-interval 1
    --tensorboard-queue-size 10
    --log-interval 1 # for terminal infos
    --save-interval 500
    --eval-interval 500
    --load $CHECKPOINT_PATH_LOAD 
    --save $CHECKPOINT_PATH_SAVE
    --eval-iters 20 # sample 20 video to eval
)

TRAIN_SCRIPT=${1:-"examples/teleai/pretrain_i2v.py"}
shift
echo "Launching: $TRAIN_SCRIPT"

torchrun ${DISTRIBUTED_ARGS[@]} ${TRAIN_SCRIPT} \
    ${MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]}    \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${LORA_CFG[@]} \
    "$@"
