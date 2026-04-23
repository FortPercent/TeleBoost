#!/usr/bin/env bash
set -xe

MODEL_PATH="/user/TeleBoost/ckpts/Wan2.1-T2V-1.3B"
OUTPUT_DIR="/user/TeleBoost/data/1__3B/rl_embeddings"
INPUT_TXT="${INPUT_TXT:-/user/TeleBoost/prompts/hard_50.txt}"

cd /user/TeleBoost
export PYTHONPATH=/user/TeleBoost:${PYTHONPATH:-}

python data_preprocess/preprocess_wan_embeddings_fromlist.py \
    --wan_model_path "$MODEL_PATH" \
    --input_txt "$INPUT_TXT" \
    --output_dir "$OUTPUT_DIR"
