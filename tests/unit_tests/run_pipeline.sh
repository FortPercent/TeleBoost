#!/bin/bash
set -e

# --- 1. 配置 PYTHONPATH (和你之前的环境保持一致) ---
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/Megatron-LM
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/yuc/env/teleai_data_tool
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/qiuyang/Video-Depth-Anything/
export PYTHONPATH=$PYTHONPATH:.  # 确保能找到当前目录下的 my_utils.py

# --- 2. 配置测试模式环境变量 (可选) ---
# 如果你想测试"真实Dump数据对比"，请解开下面两行注释并填入路径
# export WAN_DPO_DATASET_DUMP_FILE="/path/to/dataset_raw_rank0.jsonl"
# export WAN_DPO_PREVAE_TENSOR_DIR="/path/to/tensor_dumps"

# 如果不设置上面两个变量，脚本默认会进入"生成合成数据(Synthetic)"的测试分支
# 这对于单纯测试 pipeline 逻辑是否跑通已经足够了。

# --- 3. 解决潜在的 Encoder Utils 找不到的问题 ---
# 如果脚本报 "encoder_compare_utils.py not found"，请取消注释下面这行并指向该文件
# export WAN_ENCODER_UTILS_PATH="/path/to/encoder_compare_utils.py"

# --- 4. 使用 torchrun 启动 ---
# --nproc_per_node=1 : 使用 1 张卡进行测试
echo ">>> 正在使用 torchrun 启动 Pipeline 测试..."
torchrun --nproc_per_node=1 \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr="127.0.0.1" \
    --master_port="29500" \
    run_dpo_dataset_pipeline_torchrun.py