#!/usr/bin/env python3
import argparse
from pathlib import Path
import torch


# ----------------------------
# Helpers
# ----------------------------
def load_trace(path: Path) -> dict:
    obj = torch.load(path, map_location="cpu")
    if not isinstance(obj, dict):
        raise RuntimeError(f"Trace top-level is not a dict: {path}")
    return obj


def get_events(obj: dict) -> list[dict]:
    if "events" in obj and isinstance(obj["events"], list):
        return obj["events"]
    raise RuntimeError("Trace missing `events` (list). Please re-dump with ordered recorder.")


def ev_name(ev: dict) -> str:
    return ev.get("module_name") or ev.get("name") or ev.get("module") or ev.get("path") or "<unknown>"


def pick_first_dict(ev: dict, keys: list[str]) -> dict | None:
    for k in keys:
        v = ev.get(k, None)
        if isinstance(v, dict):
            return v
    return None


def normalize_per_param_map(m: dict) -> dict[str, str]:
    """
    Accept formats:
      - {param_name: "md5hex"}
      - {param_name: {"md5": "...", ...}}
      - {param_name: {"sha256": "...", ...}}  (fallback)
    Return:
      - {param_name: md5_or_hash_str}
    """
    out = {}
    for name, v in m.items():
        if isinstance(v, str):
            out[name] = v
        elif isinstance(v, dict):
            if "md5" in v and isinstance(v["md5"], str):
                out[name] = v["md5"]
            elif "sha256" in v and isinstance(v["sha256"], str):
                out[name] = v["sha256"]
            elif "hash" in v and isinstance(v["hash"], str):
                out[name] = v["hash"]
            else:
                # best-effort stringify
                out[name] = str(v)
        else:
            out[name] = str(v)
    return out


def print_event_param_md5(
    tag: str,
    ev: dict,
    param_map: dict[str, str] | None,
    buffer_map: dict[str, str] | None,
    *,
    indent: str = "  ",
):
    idx = ev.get("idx", None)
    name = ev_name(ev)
    print(f"[{tag}] idx={idx} module={name}")

    if param_map is None:
        print(f"{indent}(no per-param md5 map found in this event)")
    else:
        for pn in sorted(param_map.keys()):
            print(f"{indent}P {pn}: {param_map[pn]}")

    if buffer_map is not None:
        for bn in sorted(buffer_map.keys()):
            print(f"{indent}B {bn}: {buffer_map[bn]}")


# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser(description="Print per-param md5 from two forward-trace .pt files.")
    ap.add_argument("--teletron", required=True, help="e.g. /tmp/teletron_vae_trace.pt")
    ap.add_argument("--diffsynth", required=True, help="e.g. /tmp/diffsynth_vae_trace.pt")

    ap.add_argument("--mode", choices=["all", "diff"], default="all",
                    help="all: print everything; diff: only print params/buffers whose hashes differ")
    ap.add_argument("--limit-events", type=int, default=0,
                    help="0 = no limit; otherwise only check first N aligned events (by position).")
    ap.add_argument("--show-missing", action="store_true",
                    help="In diff mode, also print when one side is missing a param/buffer entry.")
    args = ap.parse_args()

    t_obj = load_trace(Path(args.teletron))
    d_obj = load_trace(Path(args.diffsynth))
    t_events = get_events(t_obj)
    d_events = get_events(d_obj)

    n = min(len(t_events), len(d_events))
    if args.limit_events and args.limit_events > 0:
        n = min(n, args.limit_events)

    # keys we will try
    PARAM_KEYS = [
        "param_md5_per_param",
        "params_md5_per_param",
        "param_md5_map",
        "param_hash_per_param",
        "param_hash_map",
    ]
    BUF_KEYS = [
        "buffer_md5_per_buffer",
        "buffers_md5_per_buffer",
        "buffer_md5_map",
        "buffer_hash_per_buffer",
        "buffer_hash_map",
    ]

    print(f"[info] teletron_events={len(t_events)} diffsynth_events={len(d_events)} aligned={n} mode={args.mode}")

    for i in range(n):
        t_ev = t_events[i]
        d_ev = d_events[i]

        t_param_raw = pick_first_dict(t_ev, PARAM_KEYS)
        d_param_raw = pick_first_dict(d_ev, PARAM_KEYS)
        t_buf_raw = pick_first_dict(t_ev, BUF_KEYS)
        d_buf_raw = pick_first_dict(d_ev, BUF_KEYS)

        t_param = normalize_per_param_map(t_param_raw) if t_param_raw else None
        d_param = normalize_per_param_map(d_param_raw) if d_param_raw else None
        t_buf = normalize_per_param_map(t_buf_raw) if t_buf_raw else None
        d_buf = normalize_per_param_map(d_buf_raw) if d_buf_raw else None

        if args.mode == "all":
            print_event_param_md5("teletron", t_ev, t_param, t_buf)
            print_event_param_md5("diffsynth", d_ev, d_param, d_buf)
            print("-" * 80)
            continue

        # diff mode
        any_diff = False
        tname = ev_name(t_ev)
        dname = ev_name(d_ev)
        header_printed = False

        def ensure_header():
            nonlocal header_printed
            if not header_printed:
                print(f"[DIFF] aligned_pos={i} teletron(idx={t_ev.get('idx')})={tname}  |  diffsynth(idx={d_ev.get('idx')})={dname}")
                header_printed = True

        # compare params
        if t_param is None or d_param is None:
            if args.show_missing and (t_param is not None or d_param is not None):
                ensure_header()
                print("  (missing per-param map on one side)")
                print(f"  teletron_has={t_param is not None} diffsynth_has={d_param is not None}")
                any_diff = True
        else:
            keys = sorted(set(t_param.keys()) | set(d_param.keys()))
            for k in keys:
                tv = t_param.get(k, None)
                dv = d_param.get(k, None)
                if tv is None or dv is None:
                    if args.show_missing:
                        ensure_header()
                        print(f"  P {k}: teletron={tv} diffsynth={dv}  (missing)")
                        any_diff = True
                    continue
                if tv != dv:
                    ensure_header()
                    print(f"  P {k}: teletron={tv} diffsynth={dv}")
                    any_diff = True

        # compare buffers
        if t_buf is None or d_buf is None:
            if args.show_missing and (t_buf is not None or d_buf is not None):
                ensure_header()
                print("  (missing per-buffer map on one side)")
                print(f"  teletron_has={t_buf is not None} diffsynth_has={d_buf is not None}")
                any_diff = True
        else:
            keys = sorted(set(t_buf.keys()) | set(d_buf.keys()))
            for k in keys:
                tv = t_buf.get(k, None)
                dv = d_buf.get(k, None)
                if tv is None or dv is None:
                    if args.show_missing:
                        ensure_header()
                        print(f"  B {k}: teletron={tv} diffsynth={dv}  (missing)")
                        any_diff = True
                    continue
                if tv != dv:
                    ensure_header()
                    print(f"  B {k}: teletron={tv} diffsynth={dv}")
                    any_diff = True

        if any_diff:
            print("-" * 80)

    print("[done]")


if __name__ == "__main__":
    main()
