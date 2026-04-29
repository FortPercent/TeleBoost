"""TeleBoost backport of `verl.utils.device.get_device_id` (not in v0.4.0)."""
from __future__ import annotations


def get_device_id() -> int:
    """Return current cuda device index. Mirrors upstream's later API."""
    import torch
    return torch.cuda.current_device()
