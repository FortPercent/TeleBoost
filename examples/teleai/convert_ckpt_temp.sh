export PYTHONPATH=
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yuc/teletron-wan/Megatron_wxe
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/vast
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/yxy/code/teleai_data_tool/
# 
TP=1
PP=1

HF_CKPT_PATH="/nvfile-heatstorage/model_zoo/huggingface/Wan2.1-I2V-14B-720P-Diffusers/transformer"
CHECKPOINT_PATH=/nvfile-heatstorage/myk/vast/teletron/step15000/
TARGET_CKPT_PATH=/nvfile-heatstorage/myk/Teletron/checkpoint/f1fn2v_1.3B

python  examples/teleai/convert_checkpoint_temp.py  \
    --hf-ckpt-path ${HF_CKPT_PATH} \
    --load ${CHECKPOINT_PATH} \
    --save ${TARGET_CKPT_PATH} \
    --target-params-dtype bf16 \
    --num-layers 30 \
    --folder-name release \
    --target-tensor-model-parallel-size ${TP} \
    --target-pipeline-model-parallel-size ${PP} \


