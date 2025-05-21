#!/bin/bash

# CUDA_VISIBLE_DEVICES=0 \
#   MASTER_PORT=12395 WORLD_SIZE=1 \
#   WORLD_SIZE=1 RANK=0 \
#   bash examples/vast/run_unified_sanity_check.sh 1 1 &
# pid1=$!
CUDA_VISIBLE_DEVICES=0 MASTER_PORT=12395 bash examples/vast/run_unified_sanity_check.sh 1 1
# CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=12395 bash examples/vast/run_unified_sanity_check.sh 1 2 #&
# pid1=$!

# CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=12395 bash examples/vast/run_unified.sh 1 2 9 # &
# pid1=$!