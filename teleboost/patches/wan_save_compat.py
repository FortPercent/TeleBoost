"""Wan-aware save_checkpoint compatibility shim.

Upstream verl 0.4.0 `FSDPCheckpointManager.save_checkpoint` unconditionally calls
`model_config.save_pretrained(local_path)` to write `config.json` next to the
weights. For HF transformers configs this works; for Wan models, `model.config`
is a `diffusers.configuration_utils.FrozenDict` which has no `save_pretrained`,
and the save crashes with `AttributeError`.

Pre-X3's in-tree verl fork avoided this by commenting out the line. Same idea
here: install a no-op `save_pretrained` on `FrozenDict` so the call short-circuits
without touching upstream code or recipe Workers.

The HF generation_config save path is gated by `unwrap_model.can_generate()` and
is independently shut off by Wan22DualModel.can_generate / WanModel.can_generate
returning False.
"""
from __future__ import annotations


def apply() -> None:
    try:
        from diffusers.configuration_utils import FrozenDict  # type: ignore
    except ImportError:
        return

    if hasattr(FrozenDict, "save_pretrained"):
        return

    def _save_pretrained_noop(self, *args, **kwargs):
        # Wan diffusion configs are not HF transformers configs; skip the HF dump.
        return None

    FrozenDict.save_pretrained = _save_pretrained_noop
