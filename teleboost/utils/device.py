"""TeleBoost backport of `verl.utils.device` symbols not in v0.4.0."""
from __future__ import annotations


def get_device_id() -> int:
    """Return current cuda device index. Mirrors upstream's later API."""
    import torch
    return torch.cuda.current_device()


def get_nccl_backend() -> str:
    """Return the right collective backend for the active device."""
    import torch
    if torch.cuda.is_available():
        return "nccl"
    try:
        import torch_npu  # noqa: F401
        return "hccl"
    except ImportError:
        raise RuntimeError("No available collective backend (cuda/npu).")
