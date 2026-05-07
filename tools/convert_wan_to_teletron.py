"""Convert a Wan-AI HuggingFace checkpoint into a TeleBoost --load directory.

Reads HF safetensors (single file or sharded; ``*.safetensors`` glob accepted),
renames Wan-style attention / embedding parameter names to the TeleBoost
(teleai) naming convention used by ``ParallelTeleaiModel``, and writes a
megatron-format checkpoint directory consumable by ``--load``.

Wan → TeleBoost rename rules (applied in order; only DiT keys, encoder /
VAE / CLIP weights are loaded separately by the encoder pipeline):

    .k.        -> .key.
    .q.        -> .query.
    .v.        -> .value.
    .o.        -> .out_proj.
    .norm_q.   -> .norm_query.
    .norm_k.   -> .norm_key.
    .k_img.    -> .img_key.
    .v_img.    -> .img_value.
    .norm_k_img. -> .norm_image_key.
    patch_embedding.  -> patch_emb.
    time_embedding.   -> time_emb.
    text_embedding.   -> text_emb.
    time_projection.  -> time_proj.

Output layout (megatron "release" format):

    <dst>/
      latest_checkpointed_iteration.txt   ("release")
      release/
        mp_rank_00/
          model_optim_rng.pt              { args: None, checkpoint_version: 3.0,
                                            model: <state_dict> }

Pass ``--load <dst>`` to ``pretrain_dpo_i2v.py`` / ``train_dpo.sh`` to bootstrap
training from these weights.

Examples
--------
    # Wan2.1-T2V-1.3B — single safetensors file
    python tools/convert_wan_to_teletron.py \\
        --src /models/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors \\
        --dst /ckpts/Wan2.1-T2V-1.3B-teletron

    # Wan2.2-I2V-A14B — 6 sharded safetensors under high_noise_model/
    python tools/convert_wan_to_teletron.py \\
        --src '/models/Wan2.2-I2V-A14B/high_noise_model/*.safetensors' \\
        --dst /ckpts/Wan2.2-I2V-A14B-high-teletron

    # Pre-renamed weights already in teleai naming (skip the rename step)
    python tools/convert_wan_to_teletron.py --src ... --dst ... --no-rename

Limitations
-----------
* Tensor-parallel size is fixed at 1 (``mp_rank_00``). For TP > 1 you must
  shard the resulting state_dict yourself.
* No optimizer / RNG state is populated — this is a *release* checkpoint
  for starting a fresh fine-tune, not a mid-run resume.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from collections import OrderedDict
from typing import Iterable

import torch
import safetensors.torch


_WAN_TO_TELEAI_RENAMES = [
    (re.compile(r"\.k\."), ".key."),
    (re.compile(r"\.q\."), ".query."),
    (re.compile(r"\.v\."), ".value."),
    (re.compile(r"\.o\."), ".out_proj."),
    (re.compile(r"\.norm_q\."), ".norm_query."),
    (re.compile(r"\.norm_k\."), ".norm_key."),
    (re.compile(r"\.k_img\."), ".img_key."),
    (re.compile(r"\.v_img\."), ".img_value."),
    (re.compile(r"\.norm_k_img\."), ".norm_image_key."),
    (re.compile(r"^patch_embedding\."), "patch_emb."),
    (re.compile(r"^time_embedding\."), "time_emb."),
    (re.compile(r"^text_embedding\."), "text_emb."),
    (re.compile(r"^time_projection\."), "time_proj."),
]


def _expand_paths(src: str | Iterable[str]) -> list[str]:
    """Expand a single path / glob / list of paths into a sorted file list."""
    if isinstance(src, str):
        sources = [src]
    else:
        sources = list(src)
    expanded: list[str] = []
    for s in sources:
        if any(c in s for c in "*?[]"):
            matched = sorted(glob.glob(s))
            if not matched:
                raise FileNotFoundError(f"glob {s!r} matched no files")
            expanded.extend(matched)
        else:
            if not os.path.exists(s):
                raise FileNotFoundError(s)
            expanded.append(s)
    return expanded


def load_state_dict(paths: list[str]) -> "OrderedDict[str, torch.Tensor]":
    """Load and concat state dicts from one or more .safetensors / .pt files."""
    state_dict: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for p in paths:
        if p.endswith(".safetensors"):
            with open(p, "rb") as fh:
                shard = safetensors.torch.load(fh.read())
        else:
            shard = torch.load(p, map_location="cpu", weights_only=False)
        before = len(state_dict)
        state_dict.update(shard)
        print(f"  loaded {p}: {len(shard)} keys (+{len(state_dict) - before})")
    return state_dict


def rename_wan_to_teleai(
    state_dict: "OrderedDict[str, torch.Tensor]",
) -> "OrderedDict[str, torch.Tensor]":
    """Apply Wan → teleai naming rules to every key."""
    renamed: "OrderedDict[str, torch.Tensor]" = OrderedDict()
    for key, value in state_dict.items():
        new_key = key
        for pattern, replacement in _WAN_TO_TELEAI_RENAMES:
            new_key = pattern.sub(replacement, new_key)
        renamed[new_key] = value
    return renamed


def save_teletron_release(
    state_dict: "OrderedDict[str, torch.Tensor]",
    dst: str,
    checkpoint_version: float = 3.0,
) -> str:
    """Write a megatron 'release' checkpoint directory at ``dst``.

    Returns the path to the saved ``model_optim_rng.pt`` for convenience.
    """
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(dst, "latest_checkpointed_iteration.txt"), "w") as f:
        f.write("release")
    mp_dir = os.path.join(dst, "release", "mp_rank_00")
    os.makedirs(mp_dir, exist_ok=True)
    ckpt_path = os.path.join(mp_dir, "model_optim_rng.pt")
    payload = OrderedDict(
        args=None,
        checkpoint_version=checkpoint_version,
        model=OrderedDict((k, v) for k, v in state_dict.items()),
    )
    torch.save(payload, ckpt_path)
    return ckpt_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--src",
        required=True,
        nargs="+",
        help=(
            "One or more paths / glob patterns pointing at HF safetensors "
            "or .pt files. Globs are expanded and sorted before loading."
        ),
    )
    parser.add_argument(
        "--dst",
        required=True,
        help="Output directory for the TeleBoost --load checkpoint.",
    )
    parser.add_argument(
        "--no-rename",
        action="store_true",
        help=(
            "Skip the Wan → teleai key rename step. Use this when the input "
            "weights already use teleai naming (e.g., a previous TeleBoost "
            "release checkpoint)."
        ),
    )
    args = parser.parse_args()

    paths = _expand_paths(args.src)
    print(f"loading {len(paths)} file(s)...")
    state_dict = load_state_dict(paths)
    print(f"  {len(state_dict)} total keys")

    if not args.no_rename:
        state_dict = rename_wan_to_teleai(state_dict)
        print("renamed keys: Wan → teleai naming")

    out = save_teletron_release(state_dict, args.dst)
    print(f"DONE: wrote {out}")
    print(f"      pass --load {args.dst} to load these weights")


if __name__ == "__main__":
    main()
