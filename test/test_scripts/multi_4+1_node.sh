
# cp -r /nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_13b /workspace/
# cp -r /nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_2p6b /workspace/

export WORLD_SIZE=5
export MASTER_ADDR=$GEMINI_HOST_IP_taskrole1_0
export MASTER_PORT=21456
export RANK=$GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX
export NCCL_DEBUG=INFO&&export NCCL_ALGO=RING

if [[ $RANK =~ ^(0|1|2|3)$ ]]; then
    # compute node for train
    export NUM_TRAIN_GPUS=8
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
elif [ "$RANK" -eq 4 ]; then
    # process node for data
    export NUM_TRAIN_GPUS=8
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
else
    # dirty code
    echo "Unsupported RANK value: $RANK"
    exit 1
fi


cp -r /nvfile-heatstorage/model_zoo/huggingface/Wan2.1-I2V-14B-720P-Diffusers/ /workspace/
cp -r /nvfile-heatstorage/model_zoo/Wan2___1-FLF2V-14B-480P-init/ /workspace/
cp -r /nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/ /workspace/
cd /nvfile-heatstorage/yxy/code/Teletron
pip install -r requirements.txt

bash examples/wan/run_wan_4+1_node.sh $NUM_TRAIN_GPUS
