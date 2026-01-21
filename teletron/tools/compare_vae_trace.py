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


def print_per_param_md5_diff(title: str, t_ev: dict, d_ev: dict, topk: int):
    t_map = event_param_md5_per_param(t_ev) if title == "param" else event_buffer_md5_per_buffer(t_ev)
    d_map = event_param_md5_per_param(d_ev) if title == "param" else event_buffer_md5_per_buffer(d_ev)

    if t_map is None or d_map is None:
        print(f"  (per-{title} md5 map missing; enable record_{title}_md5_per_{title}=True in hook)")
        return

    only_t, only_d, diff = diff_md5_maps(t_map, d_map)
    if only_t or only_d:
        print(f"  name mismatch: teletron_only={len(only_t)} diffsynth_only={len(only_d)}")
        if only_t:
            print(f"    teletron-only examples: {only_t[:min(10,len(only_t))]}")
        if only_d:
            print(f"    diffsynth-only examples: {only_d[:min(10,len(only_d))]}")

    if not diff:
        print(f"  per-{title} md5: OK (all shared entries match)")
        return

    print(f"  per-{title} md5: DIFF count={len(diff)} (showing top {min(topk, len(diff))})")
    for (name, a_md5, b_md5, a_dtype, b_dtype, a_shape, b_shape, a_numel, b_numel) in diff[:topk]:
        print(f"    - {name}:")
        print(f"        teletron md5={a_md5} dtype={a_dtype} shape={a_shape} numel={a_numel}")
        print(f"        diffsynth md5={b_md5} dtype={b_dtype} shape={b_shape} numel={b_numel}")


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
# Full-scan mode (md5)
# ----------------------------
def scan_md5_all(
    pairs,
    mode: str,
    strip_prefixes,
    regex_subs,
    *,
    only_mismatch: bool,
    check_param: bool,
    check_buffer: bool,
    print_per_entry_diff: bool,
    per_entry_topk: int,
):
    """
    Walk all aligned calls and compare param_md5 / buffer_md5.
    """
    param_bad = []
    buffer_bad = []

    def where_str(align_key, t_ev, d_ev):
        if mode == "strict":
            return f"pos={align_key} (t.idx={t_ev.get('idx')} d.idx={d_ev.get('idx')})"
        return f"key={align_key}"

    for (t_ev, d_ev, align_key) in pairs:
        t_raw = event_module_name(t_ev)
        d_raw = event_module_name(d_ev)
        t_name = normalize_name(t_raw, strip_prefixes, regex_subs)
        d_name = normalize_name(d_raw, strip_prefixes, regex_subs)
        where = where_str(align_key, t_ev, d_ev)
        # name mismatch is not necessarily a bug; still show in line
        name_show = f"{t_name} vs {d_name}"

        if check_param:
            t_md5 = event_param_md5(t_ev)
            d_md5 = event_param_md5(d_ev)
            ok = (t_md5 == d_md5)
            if (not only_mismatch) or (not ok):
                print(f"[param_md5] {'OK ' if ok else 'BAD'} {where}  {name_show}  tele={t_md5} diff={d_md5}")
            if not ok:
                param_bad.append((where, name_show, t_md5, d_md5, t_ev, d_ev))
                if print_per_entry_diff:
                    print("  -> per-param md5 diff:")
                    print_per_param_md5_diff("param", t_ev, d_ev, per_entry_topk)

        if check_buffer:
            t_b = event_buffer_md5(t_ev)
            d_b = event_buffer_md5(d_ev)
            ok = (t_b == d_b)
            if (not only_mismatch) or (not ok):
                print(f"[buffer_md5] {'OK ' if ok else 'BAD'} {where}  {name_show}  tele={t_b} diff={d_b}")
            if not ok:
                buffer_bad.append((where, name_show, t_b, d_b, t_ev, d_ev))
                if print_per_entry_diff:
                    print("  -> per-buffer md5 diff:")
                    print_per_param_md5_diff("buffer", t_ev, d_ev, per_entry_topk)

    print("\n==================== MD5 SCAN SUMMARY ====================")
    if check_param:
        print(f"param_md5 mismatches: {len(param_bad)}")
        if param_bad:
            w, n, tm, dm, _, _ = param_bad[0]
            print(f"  first param_md5 mismatch: {w}  {n}")
            print(f"    teletron={tm}\n    diffsynth={dm}")
    if check_buffer:
        print(f"buffer_md5 mismatches: {len(buffer_bad)}")
        if buffer_bad:
            w, n, tm, dm, _, _ = buffer_bad[0]
            print(f"  first buffer_md5 mismatch: {w}  {n}")
            print(f"    teletron={tm}\n    diffsynth={dm}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compare two ordered forward traces and find divergences (tensor/dtype/md5)."
    )
    ap.add_argument("--teletron", required=True)
    ap.add_argument("--diffsynth", required=True)
    ap.add_argument("--rtol", type=float, default=1e-5)
    ap.add_argument("--atol", type=float, default=1e-8)

    ap.add_argument("--mode", choices=["strict", "by-name"], default="strict",
                    help="strict: align by forward index; by-name: align by (module_name, call_k)")

    ap.add_argument("--strip-prefix", action="append", default=[],
                    help="Prefix to strip from module names (repeatable)")
    ap.add_argument("--regex-sub", action="append", default=[],
                    help=r"Regex substitution 'PATTERN=>REPL' (repeatable). Example: '^model\.'=>''")

    # dtype
    ap.add_argument("--check-param-dtype", action="store_true")
    ap.add_argument("--check-buffer-dtype", action="store_true")

    # md5 (first-bad mode)
    ap.add_argument("--check-param-md5", action="store_true")
    ap.add_argument("--check-buffer-md5", action="store_true")

    # NEW: full scan md5
    ap.add_argument("--scan-param-md5-all", action="store_true",
                    help="Scan ALL aligned calls and report param_md5 match/mismatch for each call.")
    ap.add_argument("--scan-buffer-md5-all", action="store_true",
                    help="Scan ALL aligned calls and report buffer_md5 match/mismatch for each call.")
    ap.add_argument("--only-md5-mismatch", action="store_true",
                    help="In md5 scan mode, only print mismatched lines.")
    ap.add_argument("--print-per-param-md5-diff", action="store_true",
                    help="In md5 scan mode, if a call mismatches, also print per-param/per-buffer md5 diff (if saved).")
    ap.add_argument("--param-topk", type=int, default=20,
                    help="For per-param/per-buffer md5 diffs, print at most top-k entries.")
    ap.add_argument("--stop-after-md5-scan", action="store_true",
                    help="If set, exit after md5 full scan (do not run tensor first-bad).")

    # tensor
    ap.add_argument("--ignore-tensors", action="store_true",
                    help="Only compare metadata (dtype/md5), skip tensor outputs.")
    ap.add_argument("--print-ok", action="store_true",
                    help="Print OK line for each aligned call (verbose).")
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

    # -------- NEW: full scan md5 --------
    if args.scan_param_md5_all or args.scan_buffer_md5_all:
        scan_md5_all(
            pairs,
            args.mode,
            strip_prefixes,
            regex_subs,
            only_mismatch=bool(args.only_md5_mismatch),
            check_param=bool(args.scan_param_md5_all),
            check_buffer=bool(args.scan_buffer_md5_all),
            print_per_entry_diff=bool(args.print_per_param_md5_diff),
            per_entry_topk=int(args.param_topk),
        )
        if args.stop_after_md5_scan:
            return

    # -------- original: first-bad tensor (and optional dtype/md5 checks) --------
    def where_str(align_key, t_ev, d_ev):
        if args.mode == "strict":
            return f"pos={align_key} (t.idx={t_ev.get('idx')} d.idx={d_ev.get('idx')})"
        return f"key={align_key}"

    for (t_ev, d_ev, align_key) in pairs:
        t_raw = event_module_name(t_ev)
        d_raw = event_module_name(d_ev)
        t_name = normalize_name(t_raw, strip_prefixes, regex_subs)
        d_name = normalize_name(d_raw, strip_prefixes, regex_subs)
        where = where_str(align_key, t_ev, d_ev)

        if args.check_param_dtype:
            ok, reason = compare_dtype_snap(event_param_dtypes(t_ev), event_param_dtypes(d_ev))
            if not ok:
                print(f"[FIRST BAD] {where}  module={t_name} vs {d_name}")
                print(f"  param_dtypes mismatch: {reason}")
                print(f"  teletron.param_dtypes={event_param_dtypes(t_ev)}")
                print(f"  diffsynth.param_dtypes={event_param_dtypes(d_ev)}")
                return

        if args.check_buffer_dtype:
            ok, reason = compare_dtype_snap(event_buffer_dtypes(t_ev), event_buffer_dtypes(d_ev))
            if not ok:
                print(f"[FIRST BAD] {where}  module={t_name} vs {d_name}")
                print(f"  buffer_dtypes mismatch: {reason}")
                print(f"  teletron.buffer_dtypes={event_buffer_dtypes(t_ev)}")
                print(f"  diffsynth.buffer_dtypes={event_buffer_dtypes(d_ev)}")
                return

        if args.check_param_md5:
            t_md5 = event_param_md5(t_ev)
            d_md5 = event_param_md5(d_ev)
            if t_md5 != d_md5:
                print(f"[FIRST BAD] {where}  module={t_name} vs {d_name}")
                print("  param_md5 mismatch detected BEFORE tensor compare.")
                print(f"  teletron.param_md5={t_md5}")
                print(f"  diffsynth.param_md5={d_md5}")
                if args.print_per_param_md5_diff:
                    print("  -> per-param md5 diff:")
                    print_per_param_md5_diff("param", t_ev, d_ev, int(args.param_topk))
                return

        if args.check_buffer_md5:
            t_b = event_buffer_md5(t_ev)
            d_b = event_buffer_md5(d_ev)
            if t_b != d_b:
                print(f"[FIRST BAD] {where}  module={t_name} vs {d_name}")
                print("  buffer_md5 mismatch detected BEFORE tensor compare.")
                print(f"  teletron.buffer_md5={t_b}")
                print(f"  diffsynth.buffer_md5={d_b}")
                if args.print_per_param_md5_diff:
                    print("  -> per-buffer md5 diff:")
                    print_per_param_md5_diff("buffer", t_ev, d_ev, int(args.param_topk))
                return

        if args.ignore_tensors:
            if args.print_ok:
                print(f"[OK ] {where}  {t_name}")
            continue

        t_out = event_outputs(t_ev)
        d_out = event_outputs(d_ev)
        if t_out is None or d_out is None:
            print(f"[FIRST BAD] {where}  module={t_name} vs {d_name}")
            print(f"  missing outputs: teletron={t_out is None} diffsynth={d_out is None}")
            return

        t_keys = set(t_out.keys())
        d_keys = set(d_out.keys())
        only_tk = sorted(t_keys - d_keys)
        only_dk = sorted(d_keys - t_keys)
        if only_tk or only_dk:
            print(f"[FIRST BAD] {where}  module={t_name} vs {d_name}")
            print("  tensor-key mismatch:")
            print(f"    teletron-only keys (first 20): {only_tk[:20]}")
            print(f"    diffsynth-only keys (first 20): {only_dk[:20]}")
            return

        mism = []
        for k in sorted(t_keys):
            res = compare_tensors(t_out[k], d_out[k], args.rtol, args.atol)
            if not res["ok"]:
                mism.append((k, res))

        if not mism:
            if args.print_ok:
                print(f"[OK ] {where}  {t_name}")
            continue

        print(f"[FIRST BAD] {where}  module={t_name} vs {d_name}")

        def score(item):
            _, r = item
            if "reason" in r:
                return float("inf")
            return r.get("max_abs", 0.0)

        mism_sorted = sorted(mism, key=score, reverse=True)
        k0, r0 = mism_sorted[0]
        if "reason" in r0:
            print(f"  tensor_key={k0}  {r0['reason']}")
        else:
            print(f"  tensor_key={k0}  max_abs={r0['max_abs']:.6g} mean_abs={r0['mean_abs']:.6g} "
                  f"dtype=({r0['a_dtype']},{r0['b_dtype']})")

        topk = min(args.topk, len(mism_sorted))
        if topk > 1:
            print(f"\n  [top {topk} mismatched tensors in this call]")
            for kk, rr in mism_sorted[:topk]:
                if "reason" in rr:
                    print(f"    - {kk}: {rr['reason']}")
                else:
                    print(f"    - {kk}: max_abs={rr['max_abs']:.6g} mean_abs={rr['mean_abs']:.6g} "
                          f"dtype=({rr['a_dtype']},{rr['b_dtype']})")
        return

    print("[done] No divergence found within aligned pairs.")
    if args.mode == "by-name":
        if only_t:
            print(f"[by-name] teletron-only aligned keys exist (count={len(only_t)}).")
        if only_d:
            print(f"[by-name] diffsynth-only aligned keys exist (count={len(only_d)}).")


if __name__ == "__main__":
    main()
