#!/bin/bash
set -euo pipefail

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"

export PYTHONPATH="${PYTHONPATH}:/nvfile-heatstorage/ai_infra/code/lit117/Megatron-LM"
export PYTHONPATH="${PYTHONPATH}:/nvfile-heatstorage/ai_infra/code/lit117/yuc/env/teleai_data_tool"
export PYTHONPATH="${PYTHONPATH}:/nvfile-heatstorage/ai_infra/code/lit117/qiuyang/Video-Depth-Anything/"

torchrun --standalone --nproc_per_node "${NPROC_PER_NODE}" -m pytest \
    tests/unit_tests/test_dpo_dataset_pipeline_torchrun.py \
    -o log_cli=true -o log_cli_level=INFO \
    "$@"
