#!/bin/bash

# ==========================================================
# [!] [!] [!]  请修改为你存放 .vsix 文件的目录  [!] [!] [!]
#
VSIX_DIR="/nvfile-heatstorage/tele_data_share/public/vsixs"
#
# ==========================================================

# cp -r /nvfile-heatstorage/tele_data_share/public/.vscode-server/ /root

# --- 1. Python & System Setup ---
set -e
echo "🚀 Phase 1: Installing Python packages..."
pip install -e .
pip install func_timeout
pip install nvitop
pip install -e /nvfile-heatstorage/ai_infra/code/lit117/qiuyang/diffusers-main
pip install tensordict
pip install etcd3
pip install -e my_utils/