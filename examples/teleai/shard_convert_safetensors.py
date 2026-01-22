#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from pathlib import Path
from collections import OrderedDict

import torch
from safetensors.torch import load_file


# ---------------------------
# Sharded safetensors resolver
# ---------------------------
def _find_index_json(src_dir: Path) -> Path | None:
    """
    Try to find an index json inside src_dir.
    Common names:
      - model.safetensors.index.json (HF)
      - diffusion_pytorch_model.safetensors.index.json (diffusers sometimes)
    """
    candidates = [
        src_dir / "model.safetensors.index.json",
        src_dir / "diffusion_pytorch_model.safetensors.index.json",
    ]
    for c in candidates:
        if c.exists():
            return c

    # fallback: any *.safetensors.index.json
    matches = sorted(src_dir.glob("*.safetensors.index.json"))
    if matches:
        return matches[0]
    return None


def resolve_safetensors_shards(src: str) -> list[str]:
    """
    src can be:
      - a directory containing sharded safetensors
      - an .index.json file
      - a single .safetensors file

    Returns: list of shard paths (strings), sorted.
    """
    p = Path(src)

    if p.is_dir():
        idx = _find_index_json(p)
        if idx is not None:
            p = idx
        else:
            # no index json; fallback: load all safetensors under directory
            shards = sorted(p.glob("*.safetensors"))
            if not shards:
                raise FileNotFoundError(f"No *.safetensors found under directory: {src}")
            return [str(x) for x in shards]

    # index json case
    if p.name.endswith(".safetensors.index.json"):
        data = json.loads(p.read_text(encoding="utf-8"))
        weight_map = data.get("weight_map", None)
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(f"Invalid index json (missing/empty weight_map): {p}")
        shard_files = sorted(set(weight_map.values()))
        return [str(p.parent / sf) for sf in shard_files]

    # single safetensors
    if p.name.endswith(".safetensors"):
        if not p.exists():
            raise FileNotFoundError(f"Safetensors file not found: {p}")
        return [str(p)]

    raise ValueError(f"Unsupported src: {src} (expect dir / *.safetensors / *.safetensors.index.json)")


# ---------------------------
# Loading / merging
# ---------------------------
def load_merged_state_dict_from_shards(shard_paths: list[str]) -> OrderedDict:
    """
    Load all shards and merge into a single OrderedDict.
    """
    merged = OrderedDict()
    for i, sp in enumerate(shard_paths):
        spath = Path(sp)
        if not spath.exists():
            raise FileNotFoundError(f"Shard not found: {sp}")
        sd = load_file(str(spath), device="cpu")
        # detect accidental key collisions
        for k in sd.keys():
            if k in merged:
                raise KeyError(f"Key collision while merging shards: {k} (from {sp})")
        merged.update(sd)
        print(f"[{i+1}/{len(shard_paths)}] loaded shard: {spath.name}  (+{len(sd)} tensors)")
    return merged


# ---------------------------
# Key rewrite logic (your original)
# ---------------------------
def update_state_dict_keys_wan_to_teletron(state_dict: dict) -> OrderedDict:
    output_state_dict = OrderedDict()
    replacement_rules = [
        (r'\.k\.', '.key.'),
        (r'\.q\.', '.query.'),
        (r'\.v\.', '.value.'),
        (r'\.o\.', '.out_proj.'),
        (r'\.norm_q\.', '.norm_query.'),
        (r'\.norm_k\.', '.norm_key.'),
        (r'\.k_img\.', '.img_key.'),
        (r'\.v_img\.', '.img_value.'),
        (r'\.norm_k_img\.', '.norm_image_key.'),
        (r'^patch_embedding\.', 'patch_emb.'),
        (r'^time_embedding\.', 'time_emb.'),
        (r'^text_embedding\.', 'text_emb.'),
        (r'^time_projection\.', 'time_proj.'),
    ]
    for key, value in state_dict.items():
        new_key = key
        for old, new in replacement_rules:
            new_key = re.sub(old, new, new_key)
        output_state_dict[new_key] = value
    return output_state_dict


# ---------------------------
# Save teletron release (your original, unchanged)
# ---------------------------
def save_teletron_release(state_dict: dict, checkpoint_dir: str):
    os.makedirs(checkpoint_dir, exist_ok=True)

    latest_file = os.path.join(checkpoint_dir, "latest_checkpointed_iteration.txt")
    with open(latest_file, "w", encoding="utf-8") as f:
        f.write("release")

    release_dir = os.path.join(checkpoint_dir, "release")
    os.makedirs(release_dir, exist_ok=True)

    mp_dir = os.path.join(release_dir, "mp_rank_00")
    os.makedirs(mp_dir, exist_ok=True)

    checkpoint_name = "model_optim_rng.pt"
    checkpoint_path = os.path.join(mp_dir, checkpoint_name)

    output_state_dict = OrderedDict({
        "args": None,
        "checkpoint_version": 3.0,
        "model": {k: v for k, v in state_dict.items()},
    })
    torch.save(output_state_dict, checkpoint_path)
    print(f"[OK] saved teletron checkpoint: {checkpoint_path}")


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Load sharded diffusion safetensors from a directory (or index.json), optionally rewrite keys, and save in Teletron release format."
    )
    parser.add_argument("--src", required=True, help="Source directory / .safetensors / .safetensors.index.json")
    parser.add_argument("--dst", required=True, help="Target directory to save teletron release")
    parser.add_argument("--no-rename", action="store_true", help="Do not rewrite keys (skip wan->teletron rename)")
    args = parser.parse_args()

    shard_paths = resolve_safetensors_shards(args.src)
    print(f"[INFO] resolved {len(shard_paths)} shard(s). Example:")
    for sp in shard_paths[:5]:
        print(f"  - {sp}")
    if len(shard_paths) > 5:
        print(f"  ... ({len(shard_paths)-5} more)")

    # load + merge
    state_dict = load_merged_state_dict_from_shards(shard_paths)
    print(f"[INFO] merged tensors: {len(state_dict)}")

    # optional rename
    if not args.no_rename:
        state_dict = update_state_dict_keys_wan_to_teletron(state_dict)
        print("[OK] keys renamed (wan -> teletron)")

    # save
    save_teletron_release(state_dict, args.dst)
    print(f"[DONE] converted from {args.src} -> {args.dst}")


if __name__ == "__main__":
    main()
