#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="/nvfile-heatstorage/ai_infra/code/fanyk1/yp/Teletron-dpo/"

echo "[1/4] cd ${REPO_DIR}"
cd "${REPO_DIR}"

echo "[2/4] Upgrade pip toolchain"
python -m pip install -U pip setuptools wheel

echo "[3/4] Install python deps"
pip install func_timeout diffusers==0.34.0 etcd3 tensordict

echo "[4/4] Install repo + utils"
python setup.py install
pip install -e my_utils

echo "[DONE] environment setup completed."
