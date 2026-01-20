#!/bin/bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

export PYTHONPATH="${PYTHONPATH:-}:/nvfile-heatstorage/ai_infra/code/lit117/Megatron-LM"
export PYTHONPATH="${PYTHONPATH}:/nvfile-heatstorage/ai_infra/code/lit117/yuc/env/teleai_data_tool"
export PYTHONPATH="${PYTHONPATH}:/nvfile-heatstorage/ai_infra/code/lit117/qiuyang/Video-Depth-Anything/"

export WAN_DPO_DATASET_DUMP_FILE=/nvfile-heatstorage/AIGC_H100/jiangshiqi/DiffSynth-Studio-main/dpo_dumps/dataset_raw.jsonl
export WAN_DPO_PREVAE_TENSOR_DIR=/nvfile-heatstorage/AIGC_H100/jiangshiqi/DiffSynth-Studio-main/dpo_dumps

# export WAN_DPO_DATASET_BASE_PATH="${WAN_DPO_DATASET_BASE_PATH:-}"

torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" \
    tests/unit_tests/run_dpo_dataset_pipeline_torchrun.py \
    "$@"
