"""Inject diffusion-aware Ulysses helpers into `verl.utils.ulysses`.

Pre-X3's in-tree verl/utils/ulysses.py exposed Wan-specific helpers
(`set/get_target_len`, `set/get_pad_size`, `diffusion_gather_outpus_and_unpad`,
`diffusion_slice_input_tensor_pad`, etc.). After X3 dropped that fork, callers
that still write `from verl.utils.ulysses import get_target_len` (e.g.
`teleboost.models.transformers.wan.ulysses_self_flash_attn_forward`) need those
symbols to exist on the upstream namespace.

Mirror the pre-X3 surface by attribute-injecting from
`teleboost.utils.diffusion_ulysses` onto `verl.utils.ulysses`. The functions
themselves live in teleboost; this patch is just the "make `import` work" shim.
"""
from __future__ import annotations


def apply() -> None:
    import verl.utils.ulysses as _u

    from teleboost.utils.diffusion_ulysses import (
        DiffusionGather,
        diffusion_gather_outpus_and_unpad,
        diffusion_slice_input_tensor_pad,
        gather_forward_split_backward,
        get_pad_size,
        get_target_len,
        set_pad_size,
        set_target_len,
        split_forward_gather_backward,
    )

    for name, value in [
        ("DiffusionGather", DiffusionGather),
        ("diffusion_gather_outpus_and_unpad", diffusion_gather_outpus_and_unpad),
        ("diffusion_slice_input_tensor_pad", diffusion_slice_input_tensor_pad),
        ("gather_forward_split_backward", gather_forward_split_backward),
        ("get_pad_size", get_pad_size),
        ("get_target_len", get_target_len),
        ("set_pad_size", set_pad_size),
        ("set_target_len", set_target_len),
        ("split_forward_gather_backward", split_forward_gather_backward),
    ]:
        if not hasattr(_u, name):
            setattr(_u, name, value)
