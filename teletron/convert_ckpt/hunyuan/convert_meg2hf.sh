export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/teleai-infra/litian/Megatron-LM
HUGGINGFACE_CKPT_PATH="/lustre/teleinfra/HunyuanVideo"
SOURCE_CKPT_PATH="/workspace/ckpt_tp2_2040_2700test/release"
TARGET_CKPT_PATH="/workspace/ckpt_vast_2040_2700/"
TP=2
PP=1

python convert_hunyuan.py  \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --hf-ckpt-path ${HUGGINGFACE_CKPT_PATH} \
    --target-params-dtype bf16 \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \
    --convert-checkpoint-from-megatron-to-transformers