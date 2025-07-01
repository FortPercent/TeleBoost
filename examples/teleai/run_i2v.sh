#!/bin/bash

# Runs the "175B" parameter model
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export PYTHONPATH=
# export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/teleai-infra/litian/Megatron-LM
# TODO, change to your own path
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/Megatron_060
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/Teletron
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/vast
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/teleai_data_tool/
# export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/TensorWatch
# export MEMORY_SNAPSHOT=True
# export PROF_SAVE_PATH="./log_memory_0607_2"
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F"," '{print NF}')
echo '$GPUS_PER_NODE' $MASTER_ADDR $GPUS_PER_NODE

# Change for multinode config
MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
# MASTER_ADDR='127.0.0.1'
echo '$MASTER_ADDR' $MASTER_ADDR
MASTER_PORT='11321'
NNODES=${WORLD_SIZE:-'1'}
NNODES=10
#NNODES=1 # TODO
echo '$NNODES' $NNODES

NODE_RANK=${RANK:-'0'}
echo '$NODE_RANK' $NODE_RANK
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
WORLD_SIZE=64
#WORLD_SIZE=6 # TODO
echo '$WORLD_SIZE' $WORLD_SIZE


#source ./examples/wan/setup_pyenv.sh
#setup_env_and_install
# reinstall with "rm -rf .venv"


CHECKPOINT_PATH_LOAD=/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_i2v/refactor/ckpt/teletron
CHECKPOINT_PATH_SAVE=/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_i2v/refactor/expr1
mkdir -p $CHECKPOINT_PATH_SAVE

TENSORBOARD_LOGS_PATH=./logs
MERGE_FILE=/nvfile-heatstorage/teleai-infra/wxe/Megatron-LM/data/gpt_2_merge.txt
DATA_PATH=./checkpoint
TP=1
CP=2
MBS=1
GBS=$(($WORLD_SIZE*$MBS/$CP/$TP))

N_VAE=16
N_MOE=2
# N_VAE=2 # TODO
# N_MOE=1
TOTAL_MOE_NODES=$((NNODES - N_VAE // 8))
NODES_PER_MOE=$((TOTAL_MOE_NODES / N_MOE))
#REMAINDER=$((TOTAL_MOE_NODES % N_MOE))
export I_MOE=$((NODE_RANK / NODES_PER_MOE))
echo '$I_MOE' $I_MOE

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE 
    --nnodes $NNODES 
    --node_rank $NODE_RANK
    --master_addr $MASTER_ADDR 
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 25
    --hidden-size 5120        
    --num-attention-heads 40
    --seq-length 512          
    --max-position-embeddings 4096
    --tokenizer-type NullTokenizer
    --vocab-size 0
)

TRAINING_ARGS=(
    # # --debug
    # --use-cpu-initialization
    --model ParallelTeleaiModel 
    --task-type teleai_i2v
    --micro-batch-size ${MBS}
    # --global-batch-size ${GBS}
    --train-iters 10000
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
    --use-distributed-optimizer
    --recompute-num-layers 40
    --no-rope-fusion
    --distributed-timeout-minutes 60
    # --distribute-saved-activations
)

MODEL_PARALLEL_ARGS=(
    --tensor-model-parallel-size ${TP}
    --context-parallel-size ${CP}
    --distributed-vae
    --distributed-vae-world-size $N_VAE
    --consumer-models-num $N_MOE
    --moe-step-factor-list 0.0 --moe-step-factor-list 0.833 --moe-step-factor-list 1.0
    #--moe-step-factor-list 0.0 --moe-step-factor-list 1.0 # TODO
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
    --load $CHECKPOINT_PATH_LOAD 
    --save $CHECKPOINT_PATH_SAVE/node_$I_MOE
    --eval-iters 10000
    --tensorboard-dir $TENSORBOARD_LOGS_PATH 
)


# When using lora and resume train from breakpoint, need to provide a base model path
# as lora only save the adpater weight.
# Set the LOAD args in EVAL_AND_LOGGING_ARGS to your saved ckpt path.
# Training from start is no need to provide LORA_BASE_MODEL_PATH and LOAD args.
# LORA_CFG=(
#     --lora False 
#     --lora-rank 8
#     --lora-alpha 32
#     --lora-dropout 0.05
#     --lora-target-modules q,v # usage: q,v,k,o or q
#     --lora-bias none
#     --lora-task-type FEATURE_EXTRACTION
#     --lora-base-model-path # specify one if using lora
# )

# export NCCL_IB_DISABLE=1
# export NCCL_SOCKET_IFNAME=eth0
# export NCCL_IBEXT_DISABLE=1
# echo $NCCL_SOCKET_IFNAME
# echo $NCCL_IB_DISABLE
# echo $NCCL_IBEXT_DISABLE
# export NCCL_DEBUG=INFO

torchrun ${DISTRIBUTED_ARGS[@]} examples/teleai/pretrain_i2v.py \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]}    \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${LORA_CFG[@]} \
    "$@"
