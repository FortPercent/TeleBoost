HF_CKPT_PATH="/nvfile-heatstorage/model_zoo/huggingface/Wan2.1-I2V-14B-720P-Diffusers/transformer"
SOURCE_CKPT_PATH="/workspace/Wan2___1-FLF2V-14B-480P-init"
SOURCE_CKPT_PATH="/nvfile-heatstorage/yxy/ccg/vast2/work_dirs/prone10_lowerlr/models/checkpoint_epoch_1_step_5000"
TARGET_CKPT_PATH="/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/prone_wan_tp1_pp1_layer_30"
#TARGET_CKPT_PATH="/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_tp1_pp1_layer_1"
TP=1
PP=1


# pip install -e /nvfile-heatstorage/teleai-infra/wxe/vast/
# rsync -avh --ignore-existing --info=progress2 /nvfile-heatstorage/model_zoo/Wan2___1-I2V-14B-480P /workspace/
# rsync -avh --ignore-existing --info=progress2 /nvfile-heatstorage/model_zoo/Wan2___1-FLF2V-14B-480P-init /workspace/

python  convert_wan.py  \
    --hf-ckpt-path ${HF_CKPT_PATH} \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --target-params-dtype bf16 \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP}

# HF_CKPT_PATH="/nvfile-heatstorage/model_zoo/huggingface/Wan2.1-I2V-14B-720P-Diffusers/transformer"
# SOURCE_CKPT_PATH="/nvfile-heatstorage/teleai-infra/wxe/Megatron_VAST/ckpt_wan_megatron/release"
# TARGET_CKPT_PATH="/nvfile-heatstorage/teleai-infra/wxe/Megatron_VAST/ckpt_wan_hf"

# python  convert_wan.py  \
#     --hf-ckpt-path ${HF_CKPT_PATH} \
#     --load ${SOURCE_CKPT_PATH} \
#     --save ${TARGET_CKPT_PATH} \
#     --target-params-dtype bf16 \
#     --target-tensor-model-parallel-size ${TP} \
#     --target-pipeline-model-parallel-size ${PP} \
#     --convert-checkpoint-from-megatron-to-transformers


