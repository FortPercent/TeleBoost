<<<<<<< HEAD
MODEL_PATH="/root/Wan2___1-T2V-14B"
=======
MODEL_PATH="/gemini/space/Wan2___1-T2V-14B"
>>>>>>> origin/verl
OUTPUT_DIR="/gemini/space/wuxuaner/Dancegrpo/data/14B/rl_embeddings"
export PYTHONPATH=/nvfile-heatstorage/teleai-infra/wxe/Dancegrpo_verl/wan:$PYTHONPATH

python examples/data_preprocess/preprocess_wan_data.py \
    --wan_model_path $MODEL_PATH \
    --input_json "/gemini/space/wyb/Dancegrpo/data/rl_embeddings/flattened_wan_results_with_prompt.json" \
    --output_dir $OUTPUT_DIR \
