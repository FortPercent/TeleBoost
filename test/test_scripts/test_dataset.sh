#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1,2,3 bash examples/wan/run_wan.sh 1 4 1
# CUDA_VISIBLE_DEVICES=0 \
#   MASTER_PORT=12395 WORLD_SIZE=1 \
#   WORLD_SIZE=1 RANK=0 \
#   bash examples/vast/run_unified_sanity_check.sh 1 1 &
# pid1=$!
# CUDA_VISIBLE_DEVICES=6 MASTER_PORT=12395 bash examples/vast/run_unified_sanity_check.sh 1 1
#CUDA_VISIBLE_DEVICES=6,7 MASTER_PORT=12395 bash examples/vast/run_unified_sanity_check.sh 1 2
#CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=12395 bash examples/vast/run_unified_sanity_check.sh 1 2 #&
# pid1=$!

# CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=12395 bash examples/vast/run_unified.sh 1 2 9 # &
# pid1=$!

# MASTER_ADDR=10.127.16.89 
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  MASTER_ADDR=10.127.16.59  MASTER_PORT=12395 \
  WORLD_SIZE=2 RANK=1 \
  bash examples/vast/run_unified.sh 1 4