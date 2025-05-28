#!/bin/bash

# Runs the "175B" parameter model
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/teleai-infra/litian/Megatron-LM
# export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/teleai-infra/litian/Teletron
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/teleai-infra/litian/Megatron-LM
# export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/Teletrons # TODO, change to your own path
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/teleai-infra/litian/vast
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/teleai-infra/litian/teleai_data_tool_source_code/


GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F"," '{print NF}')
echo '$GPUS_PER_NODE' $MASTER_ADDR $GPUS_PER_NODE

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
echo '$MASTER_ADDR'$MASTER_ADDR
MASTER_PORT='12341'
NNODES=${WORLD_SIZE:-'1'}

echo '$NNODES' $NNODES
NODE_RANK=${RANK:-'0'}
echo '$NODE_RANK' $NODE_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
echo '$WORLD_SIZE' $WORLD_SIZE

TP=$1
CP=$2
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))
export NUM_LAYERS=20
export NUM_SINGLE_LAYERS=40

# CHECKPOINT_PATH=/nvfile-heatstorage/teleai-infra/adk/Megatron_VAST/ckpt_tp${TP}_36_linearparallel_epoch1step2700
CHECKPOINT_PATH=/data02/model_zoo/origin_hunyuan_ckpt_tp1_2040_t2v
TENSORBOARD_LOGS_PATH=./logs
MERGE_FILE=/nvfile-heatstorage/teleai-infra/wxe/Megatron-LM/data/gpt_2_merge.txt
DATA_PATH=./checkpoint


DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NNODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 20
    --hidden-size 3072        
    --num-attention-heads 24
    --seq-length 512          
    --max-position-embeddings 4096
    --tokenizer-type NullTokenizer
    --vocab-size 0
)

TRAINING_ARGS=(
    --task-type t2v
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --train-iters 100000
    --weight-decay 1e-2
    --init-method-std 0.006 
    --clip-grad 0.0
    --bf16
    --lr 1e-5 
    --lr-decay-style constant
    --lr-warmup-fraction 0
    --recompute-granularity full 
    --recompute-method block 
    --use-distributed-optimizer
    --recompute-num-layers 42
    --no-rope-fusion
    --distributed-timeout-minutes 60
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --context-parallel-size ${CP}
)
DATA_ARGS=(
    --dataset-type VastDataset
    --data-path $DATA_PATH 
    --merge-file $MERGE_FILE 
    --split 949,50,1
    --dataloader-type single
    --num-workers 1
)

EVAL_AND_LOGGING_ARGS=(
    --tensorboard-queue-size 10
    --log-interval 1
    --save-interval 100
    --eval-interval 10000 
    --load $CHECKPOINT_PATH
    --save $CHECKPOINT_PATH
    --eval-iters 10000
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)

# necessary settings for multi-node training
export NCCL_IB_HCA=$(/nvfile-heatstorage/teleai-infra/look_mlxdevice.sh) #必须有，不修改，本次升级改动
echo NCCL_IB_HCA=$NCCL_IB_HCA
export NCCL_NVLS_ENABLE=0&&export NCCL_CROSS_NIC=0&&export NCCL_ALGO=RING

echo start tp${TP} cp${CP} training
torchrun ${DISTRIBUTED_ARGS[@]} examples/vast/pretrain_hunyuanvideo.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]}    \
    ${EVAL_AND_LOGGING_ARGS[@]} 
