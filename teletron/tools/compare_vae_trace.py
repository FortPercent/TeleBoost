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


def get_events(obj: dict) -> list[dict]:
    """
    Expect new recorder format: payload["events"] is a list in forward order.
    """
    if "events" in obj and isinstance(obj["events"], list):
        return obj["events"]
    raise RuntimeError(
        "This comparer requires ordered trace: top-level key `events` (list). "
        "Your pt seems to be old format (dict records). Re-dump with new recorder."
    )


def parse_regex_subs(items: list[str]) -> list[tuple[str, str]]:
    out = []
    for item in items:
        if "=>" not in item:
            raise RuntimeError(f"--regex-sub must be 'PATTERN=>REPL', got: {item}")
        pat, rep = item.split("=>", 1)
        out.append((pat, rep))
    return out


def normalize_name(name: str, strip_prefixes: list[str], regex_subs: list[tuple[str, str]]) -> str:
    for p in strip_prefixes:
        if name.startswith(p):
            name = name[len(p):]
    for pat, rep in regex_subs:
        name = re.sub(pat, rep, name)
    return name


def compare_tensors(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float):
    if a.shape != b.shape:
        return {"ok": False, "reason": f"shape {tuple(a.shape)} vs {tuple(b.shape)}"}
    # compare in fp32 for stability (this is compare-time, not trace-time)
    af = a.float()
    bf = b.float()
    diff = (af - bf).abs()
    return {
        "ok": bool(torch.allclose(af, bf, rtol=rtol, atol=atol)),
        "max_abs": float(diff.max().item()) if diff.numel() else 0.0,
        "mean_abs": float(diff.mean().item()) if diff.numel() else 0.0,
        "a_dtype": str(a.dtype),
        "b_dtype": str(b.dtype),
    }


def event_outputs(ev: dict) -> dict[str, torch.Tensor] | None:
    """
    Support both:
      - ev["outputs"] (new recorder)
      - ev["output"]  (compat)
      - ev["out"]     (compat)
    outputs are expected to be a dict key->tensor (already flattened in recorder).
    """
    x = ev.get("outputs", None)
    if x is None:
        x = ev.get("output", None)
    if x is None:
        x = ev.get("out", None)
    if x is None:
        return None
    if not isinstance(x, dict):
        return None
    # keep only tensors
    return {k: v for k, v in x.items() if torch.is_tensor(v)}


# ----------------------------
# Alignment builders
# ----------------------------
def build_strict_pairs(
    t_events: list[dict],
    d_events: list[dict],
    strip_prefixes: list[str],
    regex_subs: list[tuple[str, str]],
):
    """
    Pair events by position (forward order).
    Returns list of (t_ev, d_ev, pos_idx).
    """
    n = min(len(t_events), len(d_events))
    pairs = []
    for i in range(n):
        pairs.append((t_events[i], d_events[i], i))
    return pairs


def build_byname_pairs(
    t_events: list[dict],
    d_events: list[dict],
    strip_prefixes: list[str],
    regex_subs: list[tuple[str, str]],
):
    """
    Pair events by (normalized_module_name, kth_call_of_that_module).
    This survives one side having extra modules elsewhere.
    """
    def index_events(events: list[dict]):
        buckets = {}
        call_counter = {}
        for ev in events:
            name = ev.get("module_name") or ev.get("name") or ev.get("module") or ev.get("path")
            if not name:
                continue
            norm = normalize_name(name, strip_prefixes, regex_subs)
            k = call_counter.get(norm, 0)
            call_counter[norm] = k + 1
            buckets[(norm, k)] = ev
        return buckets

    t_map = index_events(t_events)
    d_map = index_events(d_events)

    inter_keys = sorted(set(t_map.keys()) & set(d_map.keys()))
    pairs = [(t_map[k], d_map[k], k) for k in inter_keys]
    only_t = sorted(set(t_map.keys()) - set(d_map.keys()))
    only_d = sorted(set(d_map.keys()) - set(t_map.keys()))
    return pairs, only_t, only_d


# ----------------------------
# Main compare logic
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compare two ordered forward traces (Teletron vs DiffSynth) and find the first divergence."
    )
    ap.add_argument("--teletron", required=True, help="teletron_vae_trace.pt")
    ap.add_argument("--diffsynth", required=True, help="diffsynth_vae_trace.pt")
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-8)
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--mode", choices=["strict", "by-name"], default="strict",
                    help="strict: align by forward index; by-name: align by (module_name, call_k)")

    # normalization knobs
    ap.add_argument("--strip-prefix", action="append", default=[], help="Prefix to strip from module names (repeatable)")
    ap.add_argument("--regex-sub", action="append", default=[],
                    help=r"Regex substitution 'PATTERN=>REPL' (repeatable). Example: '^model\.'=>''")

    # output options
    ap.add_argument("--only-mismatch", action="store_true", help="Only print mismatches, not OK lines")
    ap.add_argument("--print-ok-until-first-bad", action="store_true",
                    help="In strict mode, print OK lines until the first mismatch then stop.")

    args = ap.parse_args()

    t_obj = load_trace(Path(args.teletron))
    d_obj = load_trace(Path(args.diffsynth))

    t_events = get_events(t_obj)
    d_events = get_events(d_obj)

    regex_subs = parse_regex_subs(args.regex_sub)

    default_strip = [
        "teletron.vae_model.",
        "diffsynth.vae_model.",
        "teletron.vae.",
        "diffsynth.vae.",
    ]
    strip_prefixes = default_strip + list(args.strip_prefix)

    print(f"[info] teletron events={len(t_events)}  diffsynth events={len(d_events)}  mode={args.mode}")

    # ----------------------------
    # Build alignment pairs
    # ----------------------------
    only_t = []
    only_d = []
    if args.mode == "strict":
        pairs = build_strict_pairs(t_events, d_events, strip_prefixes, regex_subs)
    else:
        pairs, only_t, only_d = build_byname_pairs(t_events, d_events, strip_prefixes, regex_subs)
        print(f"[by-name] paired={len(pairs)}  teletron-only={len(only_t)}  diffsynth-only={len(only_d)}")
        if only_t:
            print("[by-name] teletron-only examples:", only_t[:10])
        if only_d:
            print("[by-name] diffsynth-only examples:", only_d[:10])

    # ----------------------------
    # Compare in execution order
    # ----------------------------
    mismatches = []
    first_bad = None

    def pretty_ev_id(ev: dict) -> str:
        idx = ev.get("idx", None)
        name = ev.get("module_name") or ev.get("name") or ev.get("module") or ev.get("path") or "<unknown>"
        return f"idx={idx} name={name}"

    for pair_idx, (t_ev, d_ev, align_key) in enumerate(pairs):
        t_name_raw = t_ev.get("module_name") or t_ev.get("name") or t_ev.get("module") or t_ev.get("path") or ""
        d_name_raw = d_ev.get("module_name") or d_ev.get("name") or d_ev.get("module") or d_ev.get("path") or ""
        t_name = normalize_name(t_name_raw, strip_prefixes, regex_subs)
        d_name = normalize_name(d_name_raw, strip_prefixes, regex_subs)

        t_out = event_outputs(t_ev)
        d_out = event_outputs(d_ev)

        # if either side has no outputs, treat as mismatch
        if t_out is None or d_out is None:
            entry = {
                "where": align_key,
                "module": f"{t_name}  vs  {d_name}",
                "reason": f"missing outputs: teletron={t_out is None} diffsynth={d_out is None}",
            }
            mismatches.append(entry)
            if first_bad is None:
                first_bad = entry
            if not args.only_mismatch:
                print(f"[BAD] {entry['where']} {entry['module']}  {entry['reason']}")
            if args.mode == "strict" and args.print_ok_until_first_bad:
                break
            continue

        # key intersection inside this module call
        t_keys = set(t_out.keys())
        d_keys = set(d_out.keys())
        inter = sorted(t_keys & d_keys)
        only_tk = sorted(t_keys - d_keys)
        only_dk = sorted(d_keys - t_keys)

        # module name mismatch in strict mode is informative (not always fatal)
        if args.mode == "strict" and t_name != d_name:
            # This often indicates insertion/deletion of modules before this point.
            # We still continue comparing tensors; the first true tensor mismatch is what we want.
            if not args.only_mismatch:
                print(f"[WARN] strict alignment name mismatch at pos={align_key}: {t_name} vs {d_name}")

        # missing tensor keys is a mismatch
        if only_tk or only_dk:
            entry = {
                "where": align_key,
                "module": f"{t_name}  vs  {d_name}",
                "reason": f"tensor-keys mismatch: teletron_only={only_tk[:5]} diffsynth_only={only_dk[:5]}",
            }
            mismatches.append(entry)
            if first_bad is None:
                first_bad = entry
            print(f"[BAD] {entry['where']} {entry['module']}  {entry['reason']}")
            if args.mode == "strict" and args.print_ok_until_first_bad:
                break
            continue

        # compare tensors for each key
        bad_in_this_call = None
        worst = None
        for k in inter:
            res = compare_tensors(t_out[k], d_out[k], args.rtol, args.atol)
            if not res["ok"]:
                bad_in_this_call = (k, res)
                # track worst by max_abs
                score = float("inf") if "reason" in res else res.get("max_abs", 0.0)
                if worst is None or score > worst[0]:
                    worst = (score, k, res)

        if bad_in_this_call is None:
            if not args.only_mismatch:
                # keep it short: print one OK line per call
                print(f"[OK ] {align_key}  {t_name}")
            continue

        # record mismatch (use worst key)
        _, worst_k, worst_res = worst
        entry = {
            "where": align_key,
            "module": f"{t_name}  vs  {d_name}",
            "tensor_key": worst_k,
            **worst_res,
        }
        mismatches.append(entry)
        if first_bad is None:
            first_bad = entry

        if "reason" in worst_res:
            print(f"[BAD] {entry['where']} {entry['module']}::{worst_k}  {worst_res['reason']}")
        else:
            print(
                f"[BAD] {entry['where']} {entry['module']}::{worst_k}  "
                f"max_abs={worst_res['max_abs']:.6g} mean_abs={worst_res['mean_abs']:.6g} "
                f"dtype=({worst_res.get('a_dtype')},{worst_res.get('b_dtype')})"
            )

        if args.mode == "strict" and args.print_ok_until_first_bad:
            break

    # ----------------------------
    # Summary
    # ----------------------------
    print("\n==================== SUMMARY ====================")
    if args.mode == "strict":
        print(f"Aligned calls compared: {min(len(t_events), len(d_events))}")
        if len(t_events) != len(d_events):
            print(f"[WARN] event count differs: teletron={len(t_events)} diffsynth={len(d_events)} "
                  f"(strict compares only the prefix)")
    else:
        print(f"Aligned calls compared: {len(pairs)}")

    print(f"Mismatched calls: {len(mismatches)}")

    if first_bad is not None:
        print("\n[first divergence]")
        if "reason" in first_bad:
            print(f"  where={first_bad['where']}  {first_bad['module']}")
            print(f"  reason: {first_bad['reason']}")
        else:
            print(f"  where={first_bad['where']}  {first_bad['module']}::{first_bad['tensor_key']}")
            print(f"  max_abs={first_bad['max_abs']:.6g} mean_abs={first_bad['mean_abs']:.6g} "
                  f"dtype=({first_bad.get('a_dtype')},{first_bad.get('b_dtype')})")

    # topk by max_abs (shape mismatch treated as inf)
    def score(e):
        if "reason" in e:
            return float("inf")
        return e.get("max_abs", 0.0)

    mism_sorted = sorted(mismatches, key=score, reverse=True)
    if mism_sorted:
        print(f"\n[top {min(args.topk, len(mism_sorted))} divergences]")
        for e in mism_sorted[: args.topk]:
            if "reason" in e:
                print(f"  where={e['where']}  {e['module']}  {e['reason']}")
            else:
                print(f"  where={e['where']}  {e['module']}::{e['tensor_key']}  "
                      f"max_abs={e['max_abs']:.6g} mean_abs={e['mean_abs']:.6g}")

    # helpful hint
    if args.mode == "strict" and first_bad is not None:
        print("\n[hint] If strict alignment shows early name mismatches before tensor mismatch, "
              "try --mode by-name to survive extra modules on one side.")


if __name__ == "__main__":
    main()
