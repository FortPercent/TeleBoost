export PYTHONUNBUFFERED=1
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NVTE_FUSED_ATTN=0
export NVTE_FLASH_ATTN=1
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export GPUS_PER_NODE=2
export MASTER_ADDR=$GEMINI_HOST_IP_taskrole1_0
export MASTER_PORT=7890

export MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
export MASTER_PORT=${MASTER_PORT:-'12345'}
export NNODES=1
export NODE_RANK=0
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/teleai-infra/litian/Megatron-LM
CHECKPOINT_PATH_SAVE=/nvfile-heatstorage/teleai-infra/kaikai/examples
# mkdir -p $CHECKPOINT_PATH_SAVE

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
)


TRAINING_ARGS=(
    --task-type wan_autoregressive
    --lr 1e-5
    --weight-decay 0.0
    --adam-beta1 0.0
    --adam-beta2 0.999
    --use-distributed-optimizer
)

MODEL_PARALLEL_ARGS=(
    --model CausalDiffusion
    --tensor-model-parallel-size 1
    --consumer-models-num 1
    --distributed-vae
    --distributed-vae-world-size 1

)

DATA_ARGS=(
    --dataset-type TensorDataset
    --dataloader-type causal
    --micro-batch-size 1
    --data-path  /nvfile-heatstorage/teleai-infra/kaikai/HumanData_subset_500/merged_videos_latents
)


EVAL_AND_LOGGING_ARGS=(
    --save $CHECKPOINT_PATH_SAVE
)


torchrun ${DISTRIBUTED_ARGS[@]} examples/wan/pretrain_causalwan.py \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]}    \
    ${EVAL_AND_LOGGING_ARGS[@]} \
    ${LORA_CFG[@]} \
    "$@"
