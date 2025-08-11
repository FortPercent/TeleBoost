#!/bin/bash

# Automatically find the checkpoint with the largest number
# CKPT_DIR="wan_experiments_14B"
# CKPT_PATH=$(ls -d $CKPT_DIR/checkpoint_model_*/model.pt 2>/dev/null | sort -V | tail -n 1)

# Check if checkpoint exists
# if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
#     echo "❌ No valid checkpoint found in $CKPT_DIR"
#     exit 1
# fi

CKPT_PATH="/nvfile-heatstorage/teleai-infra/kaikai/examples/iter_0001000/mp_rank_00/model_optim_rng.pt"
echo "✅ Using checkpoint: $CKPT_PATH"
if [ -z "$CKPT_PATH" ] || [ ! -f "$CKPT_PATH" ]; then
    echo "❌ No valid checkpoint found in $CKPT_DIR"
    exit 1
fi
# Extract step number from checkpoint path
# STEP_NUM=$(echo "$CKPT_PATH" | grep -o 'checkpoint_model_[0-9]*' | grep -o '[0-9]*')
# if [ -z "$STEP_NUM" ]; then
#     echo "❌ Could not extract step number from checkpoint path"
#     exit 1
# fi
STEP_NUM=1
echo "✅ Extracted step number: $STEP_NUM"

# Set config and data paths
CONFIG_PATH="/nvfile-heatstorage/teleai-infra/kaikai/dreamingforcing/WorldVideo/configs/self_forcing_df.yaml"
DATA_PATH="/nvfile-heatstorage/teleai-infra/kaikai/Self-Forcing-main/prompts/MovieGenVideoBench_extended.txt"

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
BASE_OUTPUT_DIR="/nvfile-heatstorage/teleai-infra/kaikai/results/Wan_long_videos_step_${STEP_NUM}"

# Delay another 90 seconds (total 180s), then start the 5s inference script
(
  CUDA_VISIBLE_DEVICES=0 python /nvfile-heatstorage/teleai-infra/kaikai/Teletron/examples/inference/Wan_inference_5s.py \
    --config_path $CONFIG_PATH \
    --output_folder $BASE_OUTPUT_DIR/self_forcing_5s \
    --checkpoint_path $CKPT_PATH \
    --data_path $DATA_PATH
) &

# Delay 90 seconds, then start the 15s inference script
(
  CUDA_VISIBLE_DEVICES=1 python /nvfile-heatstorage/teleai-infra/kaikai/Teletron/examples/inference/Wan_inference_15s.py \
    --config_path $CONFIG_PATH \
    --output_folder $BASE_OUTPUT_DIR/self_forcing_15s \
    --checkpoint_path $CKPT_PATH \
    --data_path $DATA_PATH
) &

# Immediately start the 25s inference script

(
  CUDA_VISIBLE_DEVICES=2 python /nvfile-heatstorage/teleai-infra/kaikai/Teletron/examples/inference/Wan_inference_25s.py \
    --config_path $CONFIG_PATH \
    --output_folder $BASE_OUTPUT_DIR/self_forcing_25s \
    --checkpoint_path $CKPT_PATH \
    --data_path $DATA_PATH 
)


# Wait for all background jobs to complete
wait
echo "✅ All inference jobs completed."
echo "✅ Results saved in: $BASE_OUTPUT_DIR"