MODEL_PATH="/gfs/platform/public/infra/Wan-AI/Wan2.2-T2V-A14B"
OUTPUT_DIR="./data/14B/rl_embeddings"
export PYTHONPATH=/nvfile-heatstorage/teleai-infra/wxe/Dancegrpo_verl/wan:$PYTHONPATH

python ./data_preprocess/preprocess_wan_data.py \
    --wan_model_path $MODEL_PATH \
    --input_json "./prompts/istock_2000.txt" \
    --output_dir $OUTPUT_DIR \
