export PYTHONPATH=
# TODO, change to your own path
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/Megatron_060

# Model parallelism settings
TP=1  # Tensor Parallelism
PP=1  # Pipeline Parallelism

# Path to the HuggingFace checkpoint and its transformer directory
HUGGINGFACE_CKPT_PATH="/nvfile-heatstorage/model_zoo/huggingface/hunyuan/hunyuanvideo_13b"
SOURCE_CKPT_PATH="/nvfile-heatstorage/hyc/vast/work_dirs/hunyuanvideo_i2vhy_sp2_720p_85_24fps_0512/models/checkpoint_epoch_1_step_5000/transformer_safetensor"
TARGET_CKPT_PATH="/nvfile-heatstorage/hyc/vast/work_dirs/hunyuanvideo_i2vhy_sp2_720p_85_24fps_0512/models/checkpoint_epoch_1_step_5000/teletron_debug"
rm -r $TARGET_CKPT_PATH


# Run the conversion script

### convert hunyuanvideo
python convert_hunyuanvideo.py  \
    --load ${SOURCE_CKPT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --hf-ckpt-path ${HUGGINGFACE_CKPT_PATH} \
    --target-params-dtype bf16 \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP}


### convert hunyuanvideo with wanvae
# WAN_VAE_PRETRAINED_PATH="/nvfile-heatstorage/model_zoo/huggingface/Wan2.1-I2V-14B-720P-Diffusers/vae" 
# python convert_hunyuanvideo_t2i.py  \
#     --load ${SOURCE_CKPT_PATH} \
#     --save ${TARGET_CKPT_PATH} \
#     --hf-ckpt-path ${HUGGINGFACE_CKPT_PATH} \
#     --wan_vae_pretrained_path ${WAN_VAE_PRETRAINED_PATH} \
#     --target-params-dtype bf16 \
#     --target-tensor-model-parallel-size ${TP} \
#     --target-pipeline-model-parallel-size ${PP}