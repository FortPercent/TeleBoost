# export PYTHONPATH=$(pwd)

# 设置单POD的GPU数量
export GPUS_PER_NODE=2
export WORLD_SIZE=1
export MASTER_ADDR=$GEMINI_HOST_IP_taskrole1_0
export MASTER_PORT=21456
export RANK=$GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX

# 设置分布式训练DDP所需的环境变量
export MASTER_ADDR=${MASTER_ADDR:-'127.0.0.1'}
export MASTER_PORT=${MASTER_PORT:-'12345'}
export NNODES=${WORLD_SIZE:-'1'}
export NODE_RANK=${RANK:-'0'}
export WORLD_SIZE=$(($GPUS_PER_NODE * $NNODES))

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"
# 设置NCCL的allreduce算法为RING
export NCCL_ALGO=RING
export NCCL_DEBUG=INFO
source /nvfile-heatstorage/teleai-infra/kaikai/Teletron/.venv/bin/activate
torchrun $DISTRIBUTED_ARGS /nvfile-heatstorage/teleai-infra/kaikai/Teletron/examples/wan/Wan_train.py --config_path /nvfile-heatstorage/teleai-infra/kaikai/dreamingforcing/WorldVideo/configs/self_forcing_df.yaml --no_visualize