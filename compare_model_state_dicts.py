#!/usr/bin/env python
# Usage:
#   python compare_model_state_dicts.py \
#     --diffsynth-pth /path/to/model_dumps/wan_video_vae_bfloat16.pth \
#     --teletron-pth /path/to/model_dumps/teletron_DiffSynthWanVideoVAE_bfloat16.pth
import argparse
from collections import Counter
from pathlib import Path

import torch


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        if all(torch.is_tensor(v) or isinstance(v, (int, float, str, bytes)) for v in obj.values()):
            return obj
        if "state_dict" in obj and isinstance(obj["state_dict"], dict):
            return obj["state_dict"]
        if "model" in obj and isinstance(obj["model"], dict):
            return obj["model"]
    raise RuntimeError("unsupported .pth format; expected a state_dict-like dict")


def _strip_prefix(state_dict, prefix):
    if not prefix:
        return state_dict
    if not all(isinstance(k, str) for k in state_dict.keys()):
        return state_dict
    if all(k.startswith(prefix) for k in state_dict.keys()):
        return {k[len(prefix):]: v for k, v in state_dict.items()}
    return state_dict


def _summarize_state_dict(state_dict):
    counts = Counter()
    tensor_count = 0
    non_tensor_count = 0
    for v in state_dict.values():
        if torch.is_tensor(v):
            tensor_count += 1
            counts[str(v.dtype)] += 1
        else:
            non_tensor_count += 1
    return {
        "keys": len(state_dict),
        "tensors": tensor_count,
        "non_tensors": non_tensor_count,
        "dtype_counts": dict(counts),
    }


def _compare_state_dicts(a, b, rtol, atol):
    mismatches = {
        "missing_in_b": [],
        "missing_in_a": [],
        "shape_or_dtype": [],
        "values": [],
    }
    max_abs = 0.0
    max_key = None
    common_keys = sorted(set(a.keys()) & set(b.keys()))
    for key in sorted(set(a.keys()) - set(b.keys())):
        mismatches["missing_in_b"].append(key)
    for key in sorted(set(b.keys()) - set(a.keys())):
        mismatches["missing_in_a"].append(key)
    for key in common_keys:
        va = a[key]
        vb = b[key]
        if not (torch.is_tensor(va) and torch.is_tensor(vb)):
            continue
        if va.shape != vb.shape or va.dtype != vb.dtype:
            mismatches["shape_or_dtype"].append(
                (key, f"{tuple(va.shape)} {va.dtype}", f"{tuple(vb.shape)} {vb.dtype}")
            )
            continue
        diff = (va.float().cpu() - vb.float().cpu()).abs()
        if not torch.allclose(va, vb, rtol=rtol, atol=atol):
            mismatches["values"].append(
                (key, float(diff.max()), float(diff.mean()))
            )
        if diff.numel() > 0:
            local_max = float(diff.max())
            if local_max > max_abs:
                max_abs = local_max
                max_key = key
    return mismatches, max_abs, max_key


def main():
    parser = argparse.ArgumentParser(description="Compare two model state_dict .pth files.")
    parser.add_argument("--diffsynth-pth", required=True, help="DiffSynth model_dumps .pth file.")
    parser.add_argument("--teletron-pth", required=True, help="Teletron model_dumps .pth file.")
    parser.add_argument("--strip-diffsynth-prefix", default="model.", help="Prefix to strip from DiffSynth keys.")
    parser.add_argument("--strip-teletron-prefix", default="", help="Prefix to strip from Teletron keys.")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    parser.add_argument("--max-keys", type=int, default=10, help="Max keys to print per mismatch category.")
    args = parser.parse_args()

    diffsynth_path = Path(args.diffsynth_pth)
    teletron_path = Path(args.teletron_pth)
    if not diffsynth_path.exists():
        raise RuntimeError(f"DiffSynth pth not found: {diffsynth_path}")
    if not teletron_path.exists():
        raise RuntimeError(f"Teletron pth not found: {teletron_path}")

    diffsynth_raw = _extract_state_dict(torch.load(diffsynth_path, map_location="cpu"))
    teletron_raw = _extract_state_dict(torch.load(teletron_path, map_location="cpu"))

    diffsynth_sd = _strip_prefix(diffsynth_raw, args.strip_diffsynth_prefix)
    teletron_sd = _strip_prefix(teletron_raw, args.strip_teletron_prefix)

    print(f"[diffsynth] summary={_summarize_state_dict(diffsynth_sd)}")
    print(f"[teletron] summary={_summarize_state_dict(teletron_sd)}")

    mismatches, max_abs, max_key = _compare_state_dicts(
        diffsynth_sd, teletron_sd, rtol=args.rtol, atol=args.atol
    )

    print(f"[compare] missing_in_teletron={len(mismatches['missing_in_b'])}")
    if mismatches["missing_in_b"]:
        print(f"[compare] missing_in_teletron_sample={mismatches['missing_in_b'][:args.max_keys]}")
    print(f"[compare] missing_in_diffsynth={len(mismatches['missing_in_a'])}")
    if mismatches["missing_in_a"]:
        print(f"[compare] missing_in_diffsynth_sample={mismatches['missing_in_a'][:args.max_keys]}")
    print(f"[compare] shape_or_dtype_mismatches={len(mismatches['shape_or_dtype'])}")
    if mismatches["shape_or_dtype"]:
        print(f"[compare] shape_or_dtype_sample={mismatches['shape_or_dtype'][:args.max_keys]}")
    print(f"[compare] value_mismatches={len(mismatches['values'])}")
    if mismatches["values"]:
        print(f"[compare] value_mismatch_sample={mismatches['values'][:args.max_keys]}")
    if max_key is not None:
        print(f"[compare] max_abs_diff={max_abs} key={max_key}")


if __name__ == "__main__":
    main()
