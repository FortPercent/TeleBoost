# export PYTHONPATH=
# export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yuc/teletron-wan/Megatron_wxe
# export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/vast
# export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/teleai_data_tool/
# 
TP=1
PP=1

# HF_CKPT_PATH="/nvfile-heatstorage/model_zoo/huggingface/Wan2.1-I2V-14B-720P-Diffusers/transformer"
# SOURCE_CKPT_PATH="/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/origin/wan_prone10_step5000"
# TARGET_CKPT_PATH="/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/prone_wan_tp1_pp1_layer_30_step5000"
# mkdir $TARGET_CKPT_PATH

# python  convert_light_wan.py  \
#     --hf-ckpt-path ${HF_CKPT_PATH} \
#     --load ${SOURCE_CKPT_PATH} \
#     --save ${TARGET_CKPT_PATH} \
#     --num-layers 30 \
#     --target-params-dtype bf16 \
#     --target-tensor-model-parallel-size ${TP} \
#     --target-pipeline-model-parallel-size ${PP}

HF_CKPT_PATH="/nvfile-heatstorage/model_zoo/huggingface/Wan2.1-I2V-14B-720P-Diffusers/transformer"
# SOURCE_CKPT_PATH="/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer20_moe_4_832/node_0/iter_0008000"
CHECKPOINT_PATH=/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_i2v/refactor/ckpt/origin
TARGET_CKPT_PATH="/nvfile-heatstorage/yxy/code/Teletron/debug/ckpt/wan_layer25_i2v/refactor/ckpt/teletron"

python  examples/teleai/convert_checkpoint_temp.py  \
    --hf-ckpt-path ${HF_CKPT_PATH} \
    --load ${CHECKPOINT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --target-params-dtype bf16 \
    --num-layers 25 \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \
    # --convert-checkpoint-from-megatron-to-transformers


