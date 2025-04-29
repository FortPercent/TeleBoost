#!/bin/bash
# mkdir tmp
# cp megatron/core/models/hunyuan/pipeline.py tmp/
# cp test/test_scripts/pipeline.py megatron/core/models/hunyuan/pipeline.py
# export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/teleai-infra/litian/Megatron-LM
# cp test/test_scripts/run_unified_sanity_check.sh ./
# cp hunyuanvideo/configs/hunyuanvideo_i2vhy.py tmp/
# cp test/test_scripts/hunyuanvideo_i2vhy_fakedata.py hunyuanvideo/configs/hunyuanvideo_i2vhy.py
# CUDA_VISIBLE_DEVICES=4 MASTER_PORT=11345 examples/vast/bash run_unified_sanity_check.sh 1 1 &
# pid0=$!
CUDA_VISIBLE_DEVICES=0,1 MASTER_PORT=12395 bash examples/vast/run_unified_sanity_check.sh 1 2 &
pid1=$!
CUDA_VISIBLE_DEVICES=2,3 MASTER_PORT=12335 bash examples/vast/run_unified_sanity_check.sh 2 1 &
pid2=$!
CUDA_VISIBLE_DEVICES=4,5,6,7 MASTER_PORT=12365 bash examples/vast/run_unified_sanity_check.sh 2 2 &
pid3=$!
# wait $pid0
# echo "finish tp1 cp1 $pid0"
wait $pid1 
echo "finish tp1 cp2 $pid1"
wait $pid2
echo "finish tp2 cp1 $pid2"
wait $pid3
echo "finish tp2 cp2 $pid3"
cd test/test_scripts

python verify_accuracy.py

