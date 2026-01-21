#!/usr/bin/env python
import argparse
from pathlib import Path
import re
import torch


# ----------------------------
# Trace loading helpers
# ----------------------------
def load_trace(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"Trace top-level is not a dict: {path}")
    return obj


def find_records(obj: dict):
    """
    Be robust to different recorder formats.
    We accept one of: records/trace/events/module_records/data
    """
    for k in ["records", "trace", "events", "module_records", "data"]:
        if k in obj:
            return obj[k], k
    raise RuntimeError("Cannot find records key in trace pt. Expected one of "
                       "records/trace/events/module_records/data")


def is_tensor(x):
    return torch.is_tensor(x)


def flatten_tensors(x, prefix: str, out: dict):
    """
    Recursively flatten tensors from arbitrary nested outputs.
    - tensor: store
    - list/tuple: recurse with [i]
    - dict: recurse with [key]
    """
    if is_tensor(x):
        out[prefix] = x.detach().cpu()
        return

    if isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            flatten_tensors(v, f"{prefix}[{i}]", out)
        return

    if isinstance(x, dict):
        for k, v in x.items():
            flatten_tensors(v, f"{prefix}[{k}]", out)
        return

    # ignore non-tensor leaves


def trace_to_tensor_map(obj: dict, want: str = "output") -> dict[str, torch.Tensor]:
    """
    Convert trace records to a flat dict:
      key := "{module_name}/{want}{subpath}" -> tensor

    It supports:
    1) records is dict: module_name -> record(dict)
    2) records is list: list of event dicts with module name
    """
    records, records_key = find_records(obj)

    flat = {}

    if isinstance(records, dict):
        for module_name, rec in records.items():
            if not isinstance(rec, dict):
                continue

            # try common fields
            x = rec.get(want, None) or rec.get(want + "s", None)
            if x is None:
                if want == "output":
                    x = rec.get("out", None)
                else:
                    x = rec.get("in", None)
            if x is None:
                continue

            flatten_tensors(x, f"{module_name}/{want}", flat)

    elif isinstance(records, list):
        for ev in records:
            if not isinstance(ev, dict):
                continue
            module_name = ev.get("module_name") or ev.get("name") or ev.get("module") or ev.get("path")
            if not module_name:
                continue

            x = ev.get(want, None)
            if x is None:
                if want == "output":
                    x = ev.get("out", None)
                else:
                    x = ev.get("in", None)
            if x is None:
                continue

            flatten_tensors(x, f"{module_name}/{want}", flat)

    else:
        raise RuntimeError(f"Unsupported records type: {type(records)} from key {records_key}")

    return flat


# ----------------------------
# Key normalization / matching
# ----------------------------
def normalize_key(k: str, strip_prefixes: list[str], regex_subs: list[tuple[str, str]]) -> str:
    """
    1) strip known prefixes
    2) apply regex substitutions
    """
    for p in strip_prefixes:
        if k.startswith(p):
            k = k[len(p):]
    for pat, rep in regex_subs:
        k = re.sub(pat, rep, k)
    return k


# ----------------------------
# Compare helpers
# ----------------------------
def compare_tensors(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float):
    if a.shape != b.shape:
        return {"ok": False, "reason": f"shape {tuple(a.shape)} vs {tuple(b.shape)}"}
    # compare in fp32 for stability
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    return {
        "ok": bool(torch.allclose(af, bf, rtol=rtol, atol=atol)),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
    }


def main():
    ap = argparse.ArgumentParser(description="Compare Teletron vs DiffSynth VAE per-module hook outputs.")
    ap.add_argument("--teletron", required=True, help="teletron_vae_trace.pt")
    ap.add_argument("--diffsynth", required=True, help="diffsynth_vae_trace.pt")
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--only-mismatch", action="store_true")

    # normalization knobs
    ap.add_argument("--strip-prefix", action="append", default=[], help="Prefix to strip from keys (repeatable)")
    ap.add_argument("--regex-sub", action="append", default=[],
                    help=r"Regex substitution 'PATTERN=>REPL' (repeatable). Example: '^model\.'=>''")
    args = ap.parse_args()

    t_obj = load_trace(Path(args.teletron))
    d_obj = load_trace(Path(args.diffsynth))

    # parse regex subs
    regex_subs = []
    for item in args.regex_sub:
        if "=>" not in item:
            raise RuntimeError(f"--regex-sub must be 'PATTERN=>REPL', got: {item}")
        pat, rep = item.split("=>", 1)
        regex_subs.append((pat, rep))

    # Flatten output tensors
    t_map_raw = trace_to_tensor_map(t_obj, want="output")
    d_map_raw = trace_to_tensor_map(d_obj, want="output")

    # Default prefix stripping (常见：不同 trace recorder name 不同)
    default_strip = [
        "teletron.vae_model.",
        "diffsynth.vae_model.",
        "teletron.vae.",
        "diffsynth.vae.",
    ]
    strip_prefixes = default_strip + list(args.strip_prefix)

    t_map = {normalize_key(k, strip_prefixes, regex_subs): v for k, v in t_map_raw.items()}
    d_map = {normalize_key(k, strip_prefixes, regex_subs): v for k, v in d_map_raw.items()}

    t_keys = set(t_map.keys())
    d_keys = set(d_map.keys())
    inter = sorted(t_keys & d_keys)
    only_t = sorted(t_keys - d_keys)
    only_d = sorted(d_keys - t_keys)

    print(f"[compare] teletron tensors={len(t_map)}  diffsynth tensors={len(d_map)}  intersection={len(inter)}")
    print(f"[compare] teletron-only={len(only_t)}  diffsynth-only={len(only_d)}")

    if only_t:
        print("\n[teletron-only examples]")
        for k in only_t[:20]:
            print(" ", k)
    if only_d:
        print("\n[diffsynth-only examples]")
        for k in only_d[:20]:
            print(" ", k)

    mismatches = []
    first_bad = None

    for k in inter:
        a = t_map[k]
        b = d_map[k]
        res = compare_tensors(a, b, args.rtol, args.atol)

        if not res.get("ok", False):
            entry = {"key": k, **res}
            mismatches.append(entry)
            if first_bad is None:
                first_bad = entry
            if args.only_mismatch:
                if "reason" in res:
                    print(f"[BAD] {k}  {res['reason']}")
                else:
                    print(f"[BAD] {k}  max_abs={res['max_abs']:.6g} mean_abs={res['mean_abs']:.6g}")
        else:
            if not args.only_mismatch:
                print(f"[OK ] {k}  max_abs={res['max_abs']:.6g} mean_abs={res['mean_abs']:.6g}")

    print("\n==================== SUMMARY ====================")
    print(f"Compared tensors: {len(inter)}")
    print(f"Mismatches: {len(mismatches)}")

    if first_bad is not None:
        print("\n[first mismatch]")
        if "reason" in first_bad:
            print(f"  {first_bad['key']}  {first_bad['reason']}")
        else:
            print(f"  {first_bad['key']}  max_abs={first_bad['max_abs']:.6g} mean_abs={first_bad['mean_abs']:.6g}")

    # sort by max_abs (shape mismatch treated as inf)
    def score(e):
        if "reason" in e:
            return float("inf")
        return e.get("max_abs", 0.0)

    mismatches_sorted = sorted(mismatches, key=score, reverse=True)
    if mismatches_sorted:
        print(f"\n[top {min(args.topk, len(mismatches_sorted))} mismatches]")
        for e in mismatches_sorted[: args.topk]:
            if "reason" in e:
                print(f"  {e['key']}  {e['reason']}")
            else:
                print(f"  {e['key']}  max_abs={e['max_abs']:.6g} mean_abs={e['mean_abs']:.6g}")


if __name__ == "__main__":
    main()
