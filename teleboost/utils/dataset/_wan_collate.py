"""TeleBoost wan-specific collate function (not in upstream verl@v0.4.0).

Pads variable-length tensors in a batch to the max length along dim=0,
records the original lengths under ``{key}_orig_lengths``, and returns
a flat dict of stacked tensors + non-tensors.

Used by recipe/dancegrpo dataloader path. Injected into
verl.utils.dataset.rl_dataset by teleboost.patches at startup so that
recipe code's ``from verl.utils.dataset.rl_dataset import wan_preprocessed_collate_function``
keeps working unchanged.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F


def wan_preprocessed_collate_function(data_list: list[dict]) -> dict:
    tensors: dict = defaultdict(list)
    non_tensors: dict = defaultdict(list)
    orig_lengths: dict = defaultdict(list)
    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                tensors[key].append(val)
                orig_lengths[key].append(val.shape[0])
            else:
                non_tensors[key].append(val)
    for key, val_list in tensors.items():
        max_len = max(v.shape[0] for v in val_list)
        padded = []
        for v in val_list:
            pad_len = max_len - v.shape[0]
            padded.append(F.pad(v, (0, 0, 0, pad_len), value=0.0))
        tensors[key] = torch.stack(padded, dim=0)
    for key, val_list in orig_lengths.items():
        tensors[f"{key}_orig_lengths"] = torch.tensor(val_list, dtype=torch.int)
    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)
    return {**tensors, **non_tensors}
