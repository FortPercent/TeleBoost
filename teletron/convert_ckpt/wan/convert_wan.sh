export PYTHONPATH=
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yuc/teletron-wan/Megatron_wxe
# export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/teleai-infra/wxe/Teletron
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/ccg/vast2
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/teleai-infra/wxe/teleai_data_tool/
# 

HF_CKPT_PATH="/nvfile-heatstorage/model_zoo/huggingface/Wan2.1-I2V-14B-720P-Diffusers/transformer"
SOURCE_CKPT_PATH="/workspace/Wan2___1-FLF2V-14B-480P-init"
SOURCE_CKPT_PATH="/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/prone_wan_tp1_pp1_layer_30_step100/origin"
TARGET_CKPT_PATH="/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/prone_wan_tp1_pp1_layer_30_step100"
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


