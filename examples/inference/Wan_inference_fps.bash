#!/bin/bash

# Automatically find the checkpoint with the largest number
CKPT_DIR="wan_fps_forcing_experiments_14B_bi_direct_v6"
CKPT_PATH=$(ls -d $CKPT_DIR/checkpoint_model_*/model.pt 2>/dev/null | sort -V | tail -n 1)
CKPT_PATH=/gemini/space/xxz/WorldVideo/wan_fps_forcing_experiments_14B_bi_direct_v6/checkpoint_model_008000/model.pt

# Check if checkpoint exists
if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
    echo "❌ No valid checkpoint found in $CKPT_DIR"
    exit 1
fi

echo "✅ Using checkpoint: $CKPT_PATH"

# Extract step number from checkpoint path
STEP_NUM=$(echo "$CKPT_PATH" | grep -o 'checkpoint_model_[0-9]*' | grep -o '[0-9]*')
if [ -z "$STEP_NUM" ]; then
    echo "❌ Could not extract step number from checkpoint path"
    exit 1
fi

echo "✅ Extracted step number: $STEP_NUM"

# Set config and data paths
CONFIG_PATH="configs/self_forcing_df.yaml"
DATA_PATH="prompts/MovieGenVideoBench_extended.txt"

# Check if required files exist
if [ ! -f "$CONFIG_PATH" ]; then
    echo "❌ Config file not found: $CONFIG_PATH"
    exit 1
fi

if [ ! -f "$DATA_PATH" ]; then
    echo "❌ Data file not found: $DATA_PATH"
    exit 1
fi

# Create dynamic output folder with step number
BASE_OUTPUT_DIR="outputs/Wan_14B_fps_long_videos_step_${STEP_NUM}"

echo "✅ output_folder: $BASE_OUTPUT_DIR"

# (
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python Wan_fps_inference_parallel.py \
#     --config_path $CONFIG_PATH \
#     --output_folder $BASE_OUTPUT_DIR/teacher_fps_bi_v6_parellel \
#     --checkpoint_path $CKPT_PATH \
#     --data_path $DATA_PATH
# ) &

# Delay another 90 seconds (total 180s), then start the 5s inference script
# (
#   CUDA_VISIBLE_DEVICES=0,1 python Wan_fps_inference.py \
#     --config_path $CONFIG_PATH \
#     --output_folder $BASE_OUTPUT_DIR/teacher_fps_bi_v6 \
#     --checkpoint_path $CKPT_PATH \
#     --data_path $DATA_PATH
# ) &

DATA_PATH="prompts/MovieGenVideoBench_extended_1.txt"
(
  CUDA_VISIBLE_DEVICES=0,1 python Wan_fps_inference_15s.py \
    --config_path $CONFIG_PATH \
    --output_folder $BASE_OUTPUT_DIR/outputs_ours \
    --checkpoint_path $CKPT_PATH \
    --data_path $DATA_PATH
) &

# DATA_PATH="prompts/MovieGenVideoBench_extended_2.txt"
# (
#   CUDA_VISIBLE_DEVICES=2,3 python Wan_fps_inference_15s.py \
#     --config_path $CONFIG_PATH \
#     --output_folder $BASE_OUTPUT_DIR/outputs_ours \
#     --checkpoint_path $CKPT_PATH \
#     --data_path $DATA_PATH
# ) &

# DATA_PATH="prompts/MovieGenVideoBench_extended_3.txt"
# (
#   CUDA_VISIBLE_DEVICES=4,5 python Wan_fps_inference_15s.py \
#     --config_path $CONFIG_PATH \
#     --output_folder $BASE_OUTPUT_DIR/outputs_ours \
#     --checkpoint_path $CKPT_PATH \
#     --data_path $DATA_PATH
# ) &

# DATA_PATH="prompts/MovieGenVideoBench_extended_4.txt"
# (
#   CUDA_VISIBLE_DEVICES=6,7 python Wan_fps_inference_15s.py \
#     --config_path $CONFIG_PATH \
#     --output_folder $BASE_OUTPUT_DIR/outputs_ours \
#     --checkpoint_path $CKPT_PATH \
#     --data_path $DATA_PATH
# ) &

# Wait for all background jobs to complete
wait
echo "✅ All inference jobs completed."
echo "✅ Results saved in: $BASE_OUTPUT_DIR"