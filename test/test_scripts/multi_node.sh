
# cp -r /nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_13b /workspace/
# cp -r /nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_2p6b /workspace/

export WORLD_SIZE=2
export MASTER_ADDR=$GEMINI_HOST_IP_taskrole1_0
export MASTER_PORT=12349
export RANK=$GEMINI_CURRENT_TASK_ROLE_CURRENT_TASK_INDEX
echo $WORLD_SIZE
cp -r /nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_13b /workspace/
cp -r /nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_2p6b /workspace/
cp -r /nvfile-heatstorage/model_zoo/huggingface/Wan2.1-I2V-14B-720P-Diffusers/ /workspace/
cd /nvfile-heatstorage/yxy/code/Teletron
bash examples/vast/run_hunyuant2i.sh 1 4 1