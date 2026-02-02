MODEL_PATH="/nvfile-heatstorage/ai_infra/code/wuxn5/qrl760/DanceGRPO/models/Wan2.2-T2V-A14B"
OUTPUT_DIR="/nvfile-heatstorage/ai_infra/code/wuxn5/qrl760/DanceGRPO/Dance-grpo/data/14B/rl_embeddings"
export PYTHONPATH=/nvfile-heatstorage/teleai-infra/wxe/Dancegrpo_verl/wan:$PYTHONPATH

python /nvfile-heatstorage/ai_infra/code/wuxn5/qrl760/DanceGRPO/Dance-grpo/data_preprocess/preprocess_wan_data.py \
    --wan_model_path $MODEL_PATH \
    --input_json "/nvfile-heatstorage/ai_infra/code/wuxn5/qrl760/DanceGRPO/Dance-grpo/prompts/istock_2000.txt" \
    --output_dir $OUTPUT_DIR \
