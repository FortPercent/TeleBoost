MODEL_PATH="/gfs/space/chatrl/users/wxe/Wan-AI/Wan2.1-T2V-1.3B"
OUTPUT_DIR="/gfs/space/chatrl/users/wxe/Dancegrpo-verl/data"
export PYTHONPATH=/nvfile-heatstorage/teleai-infra/wxe/Dancegrpo_verl/wan:$PYTHONPATH

python examples/data_preprocess/preprocess_wan_data.py \
    --wan_model_path $MODEL_PATH \
    --input_json "/gfs/space/chatrl/users/wxe/prompts/istock_2000.txt" \
    --output_dir $OUTPUT_DIR \
