export PYTHONPATH=
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yuc/teletron-wan/Megatron_wxe
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/vast
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/teleai_data_tool/
# 
TP=1
PP=1

HF_CKPT_PATH="/nvfile-heatstorage/model_zoo/huggingface/Wan2.1-I2V-14B-720P-Diffusers/transformer"
CHECKPOINT_PATH=/nvfile-heatstorage/ljq/repos/vast/work_dirs/wanvideo_i2v/moe_9b_720p_lownoise=937_1000/models/checkpoint_epoch_1_step_1000/transformer
TARGET_CKPT_PATH="/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_i2v/refactor/ckpt/vast_4_moe"

python  examples/teleai/convert_checkpoint_temp.py  \
    --hf-ckpt-path ${HF_CKPT_PATH} \
    --load ${CHECKPOINT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --target-params-dtype bf16 \
    --num-layers 25 \
    --folder-name node_3 \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \


