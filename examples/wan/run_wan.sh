#!/bin/bash

# Runs the "175B" parameter model
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=
# export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/teleai-infra/litian/Megatron-LM
# TODO, change to your own path
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/Megatron_060
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/Teletron
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/vast
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/teleai_data_tool/
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/TensorWatch
# export MEMORY_SNAPSHOT=True
# export PROF_SAVE_PATH="./log_memory"
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F"," '{print NF}')
echo '$GPUS_PER_NODE' $MASTER_ADDR $GPUS_PER_NODE

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
# MASTER_ADDR='127.0.0.1'
echo '$MASTER_ADDR'$MASTER_ADDR
MASTER_PORT='11111'
NNODES=${WORLD_SIZE:-'1'}

echo '$NNODES' $NNODES
NODE_RANK=${RANK:-'0'}
echo '$NODE_RANK' $NODE_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
echo '$WORLD_SIZE' $WORLD_SIZE

CHECKPOINT_PATH=/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_tp1_pp1_layer_1
TENSORBOARD_LOGS_PATH=./logs
# VOCAB_FILE=/nvfile-heatstorage/teleai-infra/wxe/Megatron-LM/data/gpt_2_vocab.json
MERGE_FILE=/nvfile-heatstorage/teleai-infra/wxe/Megatron-LM/data/gpt_2_merge.txt
DATA_PATH=./checkpoint

TP=1
CP=1
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))
# GBS=8

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NNODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 1
    --hidden-size 5120        
    --num-attention-heads 40
    --seq-length 512          
    --max-position-embeddings 4096
    --tokenizer-type NullTokenizer
    --vocab-size 0
)

TRAINING_ARGS=(
    --task-type wan_flf
    --micro-batch-size ${MBS}
    --global-batch-size ${GBS}
    --train-iters 10000
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
    --recompute-num-layers 3
    --no-rope-fusion
    --distributed-timeout-minutes 60
    # --distribute-saved-activations
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
    --save-interval 1000
    --eval-interval 10000 
    --load $CHECKPOINT_PATH 
    --save $CHECKPOINT_PATH 
    #--pretrained-checkpoint  /nvfile-heatstorage/teleai-infra/HunyuanVideo/transformer
    --eval-iters 10000
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
    # --ckpt-format torch # TODO, not support now
)


# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IBEXT_DISABLE=1
# echo $NCCL_SOCKET_IFNAME
# echo $NCCL_IB_DISABLE
# echo $NCCL_IBEXT_DISABLE
# export NCCL_DEBUG=INFO

torchrun ${DISTRIBUTED_ARGS[@]} examples/wan/pretrain_wan.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]}    \
    ${EVAL_AND_LOGGING_ARGS[@]} 