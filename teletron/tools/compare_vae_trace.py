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


def event_module_name(ev: dict) -> str:
    return ev.get("module_name") or ev.get("name") or ev.get("module") or ev.get("path") or "<unknown>"


def event_outputs(ev: dict) -> dict[str, torch.Tensor] | None:
    """
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
    return {k: v for k, v in x.items() if torch.is_tensor(v)}


def event_param_dtypes(ev: dict) -> dict | None:
    x = ev.get("param_dtypes", None)
    return x if isinstance(x, dict) else None


def event_buffer_dtypes(ev: dict) -> dict | None:
    x = ev.get("buffer_dtypes", None)
    return x if isinstance(x, dict) else None


def event_param_md5(ev: dict) -> str | None:
    x = ev.get("param_md5", None)
    return x if isinstance(x, str) or x is None else None


def event_param_md5_per_param(ev: dict) -> dict | None:
    # recorder stores dict: {param_name: {md5, shape, dtype, numel}}
    x = ev.get("param_md5_per_param", None)
    return x if isinstance(x, dict) else None


def event_buffer_md5(ev: dict) -> str | None:
    x = ev.get("buffer_md5", None)
    return x if isinstance(x, str) or x is None else None


def event_buffer_md5_per_buffer(ev: dict) -> dict | None:
    x = ev.get("buffer_md5_per_buffer", None)
    return x if isinstance(x, dict) else None


# ----------------------------
# Compare primitives
# ----------------------------
def compare_dtype_snap(a: dict | None, b: dict | None) -> tuple[bool, str]:
    """
    Return (ok, reason). If either is missing, treat as ok (optional metadata).
    """
    if a is None or b is None:
        return True, ""
    if a.get("counts", {}) != b.get("counts", {}):
        return False, f"counts {a.get('counts', {})} vs {b.get('counts', {})}"
    if a.get("num_params") != b.get("num_params"):
        return False, f"num {a.get('num_params')} vs {b.get('num_params')}"
    return True, ""


def compare_tensors(a: torch.Tensor, b: torch.Tensor, rtol: float, atol: float):
    if a.shape != b.shape:
        return {"ok": False, "reason": f"shape {tuple(a.shape)} vs {tuple(b.shape)}"}
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


def diff_md5_maps(a_map: dict | None, b_map: dict | None):
    """
    Compare per-param/per-buffer maps:
      name -> {md5, shape, dtype, numel}
    Returns (only_a, only_b, diff) where diff is list of tuples
      (name, a_md5, b_md5, a_dtype, b_dtype, a_shape, b_shape, a_numel, b_numel)
    """
    a_map = a_map or {}
    b_map = b_map or {}
    a_keys = set(a_map.keys())
    b_keys = set(b_map.keys())
    only_a = sorted(a_keys - b_keys)
    only_b = sorted(b_keys - a_keys)

    diff = []
    for k in sorted(a_keys & b_keys):
        av = a_map.get(k, {})
        bv = b_map.get(k, {})
        if av.get("md5") != bv.get("md5"):
            diff.append(
                (
                    k,
                    av.get("md5"),
                    bv.get("md5"),
                    av.get("dtype"),
                    bv.get("dtype"),
                    av.get("shape"),
                    bv.get("shape"),
                    av.get("numel"),
                    bv.get("numel"),
                )
            )
    return only_a, only_b, diff


# ----------------------------
# Alignment
# ----------------------------
def build_strict_pairs(t_events: list[dict], d_events: list[dict]):
    n = min(len(t_events), len(d_events))
    return [(t_events[i], d_events[i], i) for i in range(n)]


def build_byname_pairs(
    t_events: list[dict],
    d_events: list[dict],
    strip_prefixes: list[str],
    regex_subs: list[tuple[str, str]],
):
    """
    Pair by (normalized_module_name, kth_call_of_that_module).
    """
    def index_events(events: list[dict]):
        buckets = {}
        counter = {}
        for ev in events:
            raw = event_module_name(ev)
            norm = normalize_name(raw, strip_prefixes, regex_subs)
            k = counter.get(norm, 0)
            counter[norm] = k + 1
            buckets[(norm, k)] = ev
        return buckets

    t_map = index_events(t_events)
    d_map = index_events(d_events)
    inter = sorted(set(t_map.keys()) & set(d_map.keys()))
    pairs = [(t_map[k], d_map[k], k) for k in inter]
    only_t = sorted(set(t_map.keys()) - set(d_map.keys()))
    only_d = sorted(set(d_map.keys()) - set(t_map.keys()))
    return pairs, only_t, only_d


# ----------------------------
# Pretty printers
# ----------------------------
def _print_param_md5_report(prefix: str, t_ev: dict, d_ev: dict, topk: int, show_ok: bool):
    """
    Print per-module param md5 + per-param diffs (if present).
    """
    t_md5 = event_param_md5(t_ev)
    d_md5 = event_param_md5(d_ev)
    t_map = event_param_md5_per_param(t_ev)
    d_map = event_param_md5_per_param(d_ev)

    if t_md5 == d_md5 and not show_ok:
        return

    print(f"\n  [{prefix}] module param_md5:")
    print(f"    teletron: {t_md5}")
    print(f"    diffsynth: {d_md5}")
    if t_md5 == d_md5:
        print("    -> OK (module-level param md5 match)")

    # per-param diff if available
    if t_map is None or d_map is None:
        print("    (per-param md5 map missing on one side; enable record_param_md5_per_param=True in hook)")
        return

    only_t, only_d, diff = diff_md5_maps(t_map, d_map)
    if only_t or only_d:
        print(f"    name mismatch: teletron_only={len(only_t)} diffsynth_only={len(only_d)}")
        if only_t:
            print(f"      teletron-only examples: {only_t[:min(10,len(only_t))]}")
        if only_d:
            print(f"      diffsynth-only examples: {only_d[:min(10,len(only_d))]}")

    if not diff:
        print("    per-param md5: OK (all shared params match)")
        return

    print(f"    per-param md5: DIFF count={len(diff)} (showing top {min(topk, len(diff))})")
    for (name, a_md5, b_md5, a_dtype, b_dtype, a_shape, b_shape, a_numel, b_numel) in diff[:topk]:
        print(f"      - {name}:")
        print(f"          teletron md5={a_md5} dtype={a_dtype} shape={a_shape} numel={a_numel}")
        print(f"          diffsynth md5={b_md5} dtype={b_dtype} shape={b_shape} numel={b_numel}")


def _print_buffer_md5_report(prefix: str, t_ev: dict, d_ev: dict, topk: int, show_ok: bool):
    t_md5 = event_buffer_md5(t_ev)
    d_md5 = event_buffer_md5(d_ev)
    t_map = event_buffer_md5_per_buffer(t_ev)
    d_map = event_buffer_md5_per_buffer(d_ev)

    if t_md5 == d_md5 and not show_ok:
        return

    print(f"\n  [{prefix}] module buffer_md5:")
    print(f"    teletron: {t_md5}")
    print(f"    diffsynth: {d_md5}")
    if t_md5 == d_md5:
        print("    -> OK (module-level buffer md5 match)")

    if t_map is None or d_map is None:
        print("    (per-buffer md5 map missing on one side; enable record_buffer_md5_per_buffer=True in hook)")
        return

    only_t, only_d, diff = diff_md5_maps(t_map, d_map)
    if only_t or only_d:
        print(f"    name mismatch: teletron_only={len(only_t)} diffsynth_only={len(only_d)}")
        if only_t:
            print(f"      teletron-only examples: {only_t[:min(10,len(only_t))]}")
        if only_d:
            print(f"      diffsynth-only examples: {only_d[:min(10,len(only_d))]}")

    if not diff:
        print("    per-buffer md5: OK (all shared buffers match)")
        return

    print(f"    per-buffer md5: DIFF count={len(diff)} (showing top {min(topk, len(diff))})")
    for (name, a_md5, b_md5, a_dtype, b_dtype, a_shape, b_shape, a_numel, b_numel) in diff[:topk]:
        print(f"      - {name}:")
        print(f"          teletron md5={a_md5} dtype={a_dtype} shape={a_shape} numel={a_numel}")
        print(f"          diffsynth md5={b_md5} dtype={b_dtype} shape={b_shape} numel={b_numel}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compare two ordered forward traces and find the first divergence in execution order."
    )
    ap.add_argument("--teletron", required=True)
    ap.add_argument("--diffsynth", required=True)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-8)

    ap.add_argument("--mode", choices=["strict", "by-name"], default="strict",
                    help="strict: align by forward index; by-name: align by (module_name, call_k)")

    # normalization knobs
    ap.add_argument("--strip-prefix", action="append", default=[],
                    help="Prefix to strip from module names (repeatable)")
    ap.add_argument("--regex-sub", action="append", default=[],
                    help=r"Regex substitution 'PATTERN=>REPL' (repeatable). Example: '^model\.'=>''")

    # what to compare
    ap.add_argument("--check-param-dtype", action="store_true",
                    help="Compare per-call param dtype snapshot before comparing tensors.")
    ap.add_argument("--check-buffer-dtype", action="store_true",
                    help="Compare per-call buffer dtype snapshot before comparing tensors.")
    ap.add_argument("--check-param-md5", action="store_true",
                    help="Compare per-call module param_md5 (and per-param md5 if present).")
    ap.add_argument("--check-buffer-md5", action="store_true",
                    help="Compare per-call module buffer_md5 (and per-buffer md5 if present).")
    ap.add_argument("--show-md5-ok", action="store_true",
                    help="Also print md5 info even when md5 matches (useful to confirm plumbing).")
    ap.add_argument("--param-topk", type=int, default=20,
                    help="For per-param md5 diffs, print at most top-k entries (in name order).")
    ap.add_argument("--ignore-tensors", action="store_true",
                    help="Only compare metadata (dtype/md5), skip tensor outputs.")

    # output controls
    ap.add_argument("--print-ok", action="store_true",
                    help="Print OK line for each aligned call (can be verbose).")
    ap.add_argument("--topk", type=int, default=20,
                    help="After first mismatch, print top-k mismatched tensor keys within that call.")

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

    if args.mode == "strict":
        pairs = build_strict_pairs(t_events, d_events)
        if len(t_events) != len(d_events):
            print(f"[warn] event count differs: teletron={len(t_events)} diffsynth={len(d_events)} "
                  f"(strict compares only the first {len(pairs)} calls)")
        only_t = only_d = []
    else:
        pairs, only_t, only_d = build_byname_pairs(t_events, d_events, strip_prefixes, regex_subs)
        print(f"[by-name] paired={len(pairs)} teletron-only={len(only_t)} diffsynth-only={len(only_d)}")
        if only_t:
            print("[by-name] teletron-only examples:", only_t[:10])
        if only_d:
            print("[by-name] diffsynth-only examples:", only_d[:10])

    def where_str(align_key, t_ev, d_ev):
        if args.mode == "strict":
            return f"pos={align_key} (t.idx={t_ev.get('idx')} d.idx={d_ev.get('idx')})"
        return f"key={align_key}"

    # walk in aligned order and stop at first divergence
    for (t_ev, d_ev, align_key) in pairs:
        t_raw = event_module_name(t_ev)
        d_raw = event_module_name(d_ev)
        t_name = normalize_name(t_raw, strip_prefixes, regex_subs)
        d_name = normalize_name(d_raw, strip_prefixes, regex_subs)
        where = where_str(align_key, t_ev, d_ev)

        # 0) dtype snapshots
        if args.check_param_dtype:
            ok, reason = compare_dtype_snap(event_param_dtypes(t_ev), event_param_dtypes(d_ev))
            if not ok:
                print(f"[FIRST BAD] {where}  module={t_name} vs {d_name}")
                print(f"  param_dtypes mismatch: {reason}")
                print(f"  teletron.param_dtypes={event_param_dtypes(t_ev)}")
                print(f"  diffsynth.param_dtypes={event_param_dtypes(d_ev)}")
                if args.check_param_md5:
                    _print_param_md5_report("param-md5", t_ev, d_ev, args.param_topk, args.show_md5_ok)
                if args.check_buffer_md5:
                    _print_buffer_md5_report("buffer-md5", t_ev, d_ev, args.param_topk, args.show_md5_ok)
                return

        if args.check_buffer_dtype:
            ok, reason = compare_dtype_snap(event_buffer_dtypes(t_ev), event_buffer_dtypes(d_ev))
            if not ok:
                print(f"[FIRST BAD] {where}  module={t_name} vs {d_name}")
                print(f"  buffer_dtypes mismatch: {reason}")
                print(f"  teletron.buffer_dtypes={event_buffer_dtypes(t_ev)}")
                print(f"  diffsynth.buffer_dtypes={event_buffer_dtypes(d_ev)}")
                if args.check_param_md5:
                    _print_param_md5_report("param-md5", t_ev, d_ev, args.param_topk, args.show_md5_ok)
                if args.check_buffer_md5:
                    _print_buffer_md5_report("buffer-md5", t_ev, d_ev, args.param_topk, args.show_md5_ok)
                return

        # 0.5) md5 snapshots (can be checked even when tensors match)
        if args.check_param_md5:
            t_md5 = event_param_md5(t_ev)
            d_md5 = event_param_md5(d_ev)
            if t_md5 != d_md5:
                print(f"[FIRST BAD] {where}  module={t_name} vs {d_name}")
                print("  param_md5 mismatch detected BEFORE tensor compare.")
                _print_param_md5_report("param-md5", t_ev, d_ev, args.param_topk, args.show_md5_ok)
                if args.check_buffer_md5:
                    _print_buffer_md5_report("buffer-md5", t_ev, d_ev, args.param_topk, args.show_md5_ok)
                return
            elif args.show_md5_ok:
                # print ok md5 line but continue
                print(f"[MD5 OK] {where}  {t_name}  param_md5={t_md5}")

        if args.check_buffer_md5:
            t_bmd5 = event_buffer_md5(t_ev)
            d_bmd5 = event_buffer_md5(d_ev)
            if t_bmd5 != d_bmd5:
                print(f"[FIRST BAD] {where}  module={t_name} vs {d_name}")
                print("  buffer_md5 mismatch detected BEFORE tensor compare.")
                _print_buffer_md5_report("buffer-md5", t_ev, d_ev, args.param_topk, args.show_md5_ok)
                if args.check_param_md5:
                    _print_param_md5_report("param-md5", t_ev, d_ev, args.param_topk, args.show_md5_ok)
                return
            elif args.show_md5_ok:
                print(f"[MD5 OK] {where}  {t_name}  buffer_md5={t_bmd5}")

        if args.ignore_tensors:
            if args.print_ok:
                print(f"[OK ] {where}  {t_name}")
            continue

        # 1) outputs exist?
        t_out = event_outputs(t_ev)
        d_out = event_outputs(d_ev)
        if t_out is None or d_out is None:
            print(f"[FIRST BAD] {where}  module={t_name} vs {d_name}")
            print(f"  missing outputs: teletron={t_out is None} diffsynth={d_out is None}")
            if args.check_param_md5:
                _print_param_md5_report("param-md5", t_ev, d_ev, args.param_topk, args.show_md5_ok)
            if args.check_buffer_md5:
                _print_buffer_md5_report("buffer-md5", t_ev, d_ev, args.param_topk, args.show_md5_ok)
            return

        # 2) key sets must match (within this aligned call)
        t_keys = set(t_out.keys())
        d_keys = set(d_out.keys())
        only_tk = sorted(t_keys - d_keys)
        only_dk = sorted(d_keys - t_keys)
        if only_tk or only_dk:
            print(f"[FIRST BAD] {where}  module={t_name} vs {d_name}")
            print(f"  tensor-key mismatch:")
            print(f"    teletron-only keys (first 20): {only_tk[:20]}")
            print(f"    diffsynth-only keys (first 20): {only_dk[:20]}")
            if args.check_param_md5:
                _print_param_md5_report("param-md5", t_ev, d_ev, args.param_topk, args.show_md5_ok)
            if args.check_buffer_md5:
                _print_buffer_md5_report("buffer-md5", t_ev, d_ev, args.param_topk, args.show_md5_ok)
            return

        # 3) compare tensors: find mismatches in this call
        mism = []
        for k in sorted(t_keys):
            res = compare_tensors(t_out[k], d_out[k], args.rtol, args.atol)
            if not res["ok"]:
                mism.append((k, res))

        if not mism:
            if args.print_ok:
                print(f"[OK ] {where}  {t_name}")
            continue

        # FIRST divergence found at this call
        print(f"[FIRST BAD] {where}  module={t_name} vs {d_name}")

        # show param/buffer md5 + per-param/per-buffer diffs (this is what you asked)
        if args.check_param_md5:
            _print_param_md5_report("param-md5", t_ev, d_ev, args.param_topk, args.show_md5_ok)
        if args.check_buffer_md5:
            _print_buffer_md5_report("buffer-md5", t_ev, d_ev, args.param_topk, args.show_md5_ok)

        # representative mismatch (worst by max_abs, shape mismatch => inf)
        def score(item):
            _, r = item
            if "reason" in r:
                return float("inf")
            return r.get("max_abs", 0.0)

        mism_sorted = sorted(mism, key=score, reverse=True)
        k0, r0 = mism_sorted[0]
        if "reason" in r0:
            print(f"\n  tensor_key={k0}  {r0['reason']}")
        else:
            print(f"\n  tensor_key={k0}  max_abs={r0['max_abs']:.6g} mean_abs={r0['mean_abs']:.6g} "
                  f"dtype=({r0['a_dtype']},{r0['b_dtype']})")

        # also print top-k mismatched tensor keys in this call
        topk = min(args.topk, len(mism_sorted))
        print(f"\n  [top {topk} mismatched tensors in this call]")
        for kk, rr in mism_sorted[:topk]:
            if "reason" in rr:
                print(f"    - {kk}: {rr['reason']}")
            else:
                print(f"    - {kk}: max_abs={rr['max_abs']:.6g} mean_abs={rr['mean_abs']:.6g} "
                      f"dtype=({rr['a_dtype']},{rr['b_dtype']})")

        # and show dtype snapshots (for debugging drift)
        if args.check_param_dtype:
            print(f"\n  teletron.param_dtypes={event_param_dtypes(t_ev)}")
            print(f"  diffsynth.param_dtypes={event_param_dtypes(d_ev)}")
        if args.check_buffer_dtype:
            print(f"\n  teletron.buffer_dtypes={event_buffer_dtypes(t_ev)}")
            print(f"  diffsynth.buffer_dtypes={event_buffer_dtypes(d_ev)}")

        return

    print("[done] No divergence found within aligned pairs.")
    if args.mode == "by-name":
        if only_t:
            print(f"[by-name] teletron-only aligned keys exist (count={len(only_t)}).")
        if only_d:
            print(f"[by-name] diffsynth-only aligned keys exist (count={len(only_d)}).")


if __name__ == "__main__":
    main()
