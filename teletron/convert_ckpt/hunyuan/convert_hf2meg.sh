export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/Megatron_VAST

# Path to the HuggingFace checkpoint and its transformer directory
CKPT_NAME="hunyuanvideo_i2vhy_token_replace"
#CKPT_NAME="hunyuanvideo_i2v_multimask"

HUGGINGFACE_CKPT_PATH="/nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_13b"
CONFIG_FILE="/nvfile-heatstorage/yxy/code/Teletron/model_paths.json"
CKPT_PATH=$(jq -r ".${CKPT_NAME}" "$CONFIG_FILE")
SOURCE_CKPT_PATH="${CKPT_PATH}/transformer_safetensor"
TARGET_CKPT_PATH="${CKPT_PATH}/teletron"

# Model parallelism settings
TP=1  # Tensor Parallelism
PP=1  # Pipeline Parallelism

rm -r $TARGET_CKPT_PATH

# Run the conversion script
python convert_hunyuanvideo.py  \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --hf-ckpt-path ${HUGGINGFACE_CKPT_PATH} \
    --target-params-dtype bf16 \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP}
