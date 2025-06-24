#!/bin/bash

setup_env_and_install() {
    [[ -d ".venv" ]] && { echo "✅ 虚拟环境已存在，跳过安装步骤..."; return 0; }
    
    echo "🔧 开始环境安装和依赖配置..."
    set -euo pipefail
    
    echo "📦 1. 开始设置环境依赖安装..."
    pip install --root-user-action ignore \
        --trusted-host pypi.chinatelecom.ai \
        --index-url http://pypi.chinatelecom.ai/simple/ uv || { echo "❌ pip安装uv失败！"; return 1; }
    
    echo "🏗️  创建新的虚拟环境..."
    uv venv --system-site-packages || { echo "❌ 创建虚拟环境失败！"; return 1; }
    source .venv/bin/activate || { echo "❌ 激活虚拟环境失败！"; return 1; }
    
    echo "✅ 2. 虚拟环境已激活"
    local projects=(
        "/nvfile-heatstorage/teleai-infra/wxy/Megatron-LM" # clean Megatron 0.6.0
        "/nvfile-heatstorage/teleai-infra/wxy/vast" # clean Vast
        "/nvfile-heatstorage/teleai-infra/wxy/teleai_data_tool" # clean teleai_data_tool
        "."
    )
    echo "📋 3. 开始安装项目依赖..."
    for project in "${projects[@]}"; do
        if [[ -d "$project" ]]; then
            echo "📦 正在安装项目: $project"
            uv pip install -e "$project" || { echo "❌ 安装项目 $project 失败！"; return 1; }
        else
            echo "⚠️  项目目录不存在，跳过: $project"
        fi
    done
    
    echo "🎉 所有依赖安装成功！"
    echo "📁 虚拟环境位置: $(pwd)/.venv"
}