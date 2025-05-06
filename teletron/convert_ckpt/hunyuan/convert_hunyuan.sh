SOURCE_CKPT_PATH="/nvfile-heatstorage/teleai-infra/litian/megatron_ckpt/ckpt_tp1_2040_linearparallel_epoch1step2700/iter_0000001/"
TARGET_CKPT_PATH="/nvfile-heatstorage/teleai-infra/wxe/temp"
TP=1
PP=1

python convert_hunyuanvideo.py  \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --target-params-dtype bf16 \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \
    --convert-checkpoint-from-megatron-to-transformers