"""TeleBoost backport of `verl.utils.model.convert_weight_keys` (not in v0.4.0).

HF transformers >=4.51 sets `_checkpoint_conversion_mapping` on some models so
state_dict keys can round-trip between HF-canonical and runtime forms. This
helper applies the reverse mapping when re-loading. Used by the diffusion
sharding manager during weight reload.
"""
from __future__ import annotations

import re
from typing import Dict


def convert_weight_keys(state_dict: Dict, model):
    if not hasattr(model, "_checkpoint_conversion_mapping"):
        return state_dict

    reverse_key_mapping = {v: k for k, v in model._checkpoint_conversion_mapping.items()}
    original_weights = {}
    for key, value in state_dict.items():
        for pattern, replacement in reverse_key_mapping.items():
            replacement = replacement.lstrip("^")
            replacement = re.sub(r"\(.*\)", "", replacement)
            key, n_replace = re.subn(pattern, replacement, key)
            if n_replace > 0:
                break
        original_weights[key] = value
    return original_weights
