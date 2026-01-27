#!/bin/bash

# --- 1. 配置 PYTHONPATH ---
# 将所需的库路径追加到现有的 PYTHONPATH 中
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/Megatron-LM
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/yuc/env/teleai_data_tool
export PYTHONPATH=$PYTHONPATH:/nvfile-heatstorage/ai_infra/code/lit117/qiuyang/Video-Depth-Anything/

# (可选) 将当前目录也加入 path，防止找不到当前目录下的模块
export PYTHONPATH=$PYTHONPATH:.

# --- 2. 打印检查 (可选，用于 Debug) ---
echo ">>> PYTHONPATH 已配置:"
echo $PYTHONPATH
echo "---------------------------------------------------"

# --- 3. 启动 Python 脚本 ---
# 假设你的测试文件名为 debug_dataset.py
echo ">>> 正在启动测试脚本 debug_dataset.py ..."
python debug_dataset.py

# 如果你想看更详细的测试过程，可以使用下面这就话代替上面那句：
# python debug_dataset.py -v