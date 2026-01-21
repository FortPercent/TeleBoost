#!/usr/bin/env python
import argparse
from pathlib import Path
import torch

def load(path: Path):
    obj = torch.load(path, map_location="cpu")
    events = obj.get("events", None)
    if not isinstance(events, list):
        raise RuntimeError("trace has no `events` list. Re-dump with new recorder.")
    return obj, events

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teletron", required=True)
    ap.add_argument("--diffsynth", required=True)
    ap.add_argument("--mode", choices=["strict"], default="strict")
    ap.add_argument("--show-buffers", action="store_true")
    args = ap.parse_args()

    t_obj, t_events = load(Path(args.teletron))
    d_obj, d_events = load(Path(args.diffsynth))

    n = min(len(t_events), len(d_events))
    print(f"[info] teletron events={len(t_events)} diffsynth events={len(d_events)} printing first {n} aligned calls")

    for i in range(n):
        te = t_events[i]
        de = d_events[i]
        tname = te.get("module_name", "<unknown>")
        dname = de.get("module_name", "<unknown>")

        tp = te.get("param_dtypes", None)
        dp = de.get("param_dtypes", None)

        print(f"\n[{i}] teletron={tname}")
        print(f"    param_dtypes={tp}")
        print(f"[{i}] diffsynth={dname}")
        print(f"    param_dtypes={dp}")

        if args.show_buffers:
            tb = te.get("buffer_dtypes", None)
            db = de.get("buffer_dtypes", None)
            print(f"    buffer_dtypes(tele)={tb}")
            print(f"    buffer_dtypes(diff)={db}")

if __name__ == "__main__":
    main()
