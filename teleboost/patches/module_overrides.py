"""Whole-module overrides over upstream verl.

For files where the project diverges deeply from upstream verl@v0.4.0
(rl_dataset.py adds wan_preprocessed_collate_function and friends, etc),
attribute injection is too cumbersome - we replace the whole module via
sys.modules. The override modules live under teleboost/_overrides/verl/...
and mirror the upstream module path.

Apply BEFORE any import that pulls verl.X.Y for an overridden Y, so the
sys.modules rewrite takes effect for downstream `from verl.X.Y import Z`.
"""
from __future__ import annotations

import importlib
import sys

# verl module path -> teleboost override module path
OVERRIDES = {
    "verl.utils.dataset.rl_dataset": "teleboost._overrides.verl.utils.dataset.rl_dataset",
    "verl.models.transformers.monkey_patch": "teleboost._overrides.verl.models.transformers.monkey_patch",
    # The wan22 / wan modules already live in teleboost.models.transformers
    # (stage 1 mv); alias them under verl namespace so monkey_patch.py's
    # `from .wan import ulysses_self_flash_attn_forward` resolves.
    "verl.models.transformers.wan": "teleboost.models.transformers.wan",
    "verl.models.transformers.wan22": "teleboost.models.transformers.wan22",
}


def apply() -> None:
    for verl_path, teleboost_path in OVERRIDES.items():
        try:
            mod = importlib.import_module(teleboost_path)
        except Exception as e:  # noqa: BLE001 - want to keep going
            print(f"[teleboost.patches] WARNING: failed to load override {teleboost_path}: {e}")
            continue
        sys.modules[verl_path] = mod
