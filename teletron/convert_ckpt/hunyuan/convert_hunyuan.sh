
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/teleai-infra/litian/Megatron-LM
HUGGINGFACE_CKPT_PATH="/nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_13b"
SOURCE_CKPT_PATH="/nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_13b/transformer"
# SOURCE_CKPT_PATH="/nvfile-heatstorage/teleai-infra/litian/megatron_ckpt/ckpt_tp1_2040_linearparallel_epoch1step2700/iter_0002400"
# TARGET_CKPT_PATH="/nvfile-heatstorage/teleai-infra/litian/megatron_ckpt/vast_ckpt/tp1_iter2400"
# SOURCE_CKPT_PATH="/nvfile-heatstorage/ljq/repos/vast/work_dirs/hunyuanvideo_i2vhy_newdataset_720p_1e5_spring_newdata_0210/models/checkpoint_epoch_1_step_2700/transformer_safetensor"
TARGET_CKPT_PATH="/workspace/ckpt_tp1_2040_origin/"
TP=1
PP=1

python convert_hunyuanvideo.py  \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --hf-ckpt-path ${HUGGINGFACE_CKPT_PATH} \
    --target-params-dtype bf16 \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} #\
    #--convert-checkpoint-from-megatron-to-transformers
    