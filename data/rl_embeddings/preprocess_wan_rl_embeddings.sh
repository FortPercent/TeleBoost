MODEL_PATH="/gemini/platform/public/zqni/wan2.1/Wan2.1-T2V-1.3B"
OUTPUT_DIR="data/rl_embeddings"

python preprocess_wan_embeddings.py \
    --wan_model_path $MODEL_PATH \
    --input_json "data/rl_embeddings/flattened_wan_results_with_prompt.json" \
    --output_dir $OUTPUT_DIR \
