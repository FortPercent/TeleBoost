
# cp -r /nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_13b /workspace/
# cp -r /nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_2p6b /workspace/

export WORLD_SIZE=2
export MASTER_ADDR=$GEMINI_HOST_IP_taskrole1_0
export MASTER_PORT=21456
export RANK=$GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX
export NCCL_DEBUG=INFO&&export NCCL_ALGO=RING
pip install -r requirements.txt
cp -r /nvfile-heatstorage/model_zoo/huggingface/Wan2.1-I2V-14B-720P-Diffusers/ /workspace/
cp -r /nvfile-heatstorage/model_zoo/Wan2___1-FLF2V-14B-480P-init/ /workspace/
cp -r /nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P/ /workspace/
cd /nvfile-heatstorage/yxy/code/Teletron
bash examples/wan/run_wan.sh
