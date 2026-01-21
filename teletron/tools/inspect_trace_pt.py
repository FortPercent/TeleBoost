#!/usr/bin/env python
import argparse
from pathlib import Path
import torch


def _is_tensor(x):
    return torch.is_tensor(x)


def _summ_tensor(t: torch.Tensor, with_stats: bool = True) -> dict:
    t_cpu = t.detach().cpu()
    d = {
        "shape": tuple(t_cpu.shape),
        "dtype": str(t_cpu.dtype),
        "device": str(t.device),
    }
    if with_stats:
        # 用 float32 统计，避免 bf16 统计溢出/精度问题
        tf = t_cpu.float()
        d.update(
            min=float(tf.min().item()) if tf.numel() else None,
            max=float(tf.max().item()) if tf.numel() else None,
            mean=float(tf.mean().item()) if tf.numel() else None,
        )
    return d


def _summ_obj(x, max_list_elems=3):
    if _is_tensor(x):
        return {"type": "tensor", **_summ_tensor(x)}
    if isinstance(x, (int, float, str, bool)) or x is None:
        return {"type": type(x).__name__, "value": x}
    if isinstance(x, (list, tuple)):
        return {
            "type": type(x).__name__,
            "len": len(x),
            "head": [_summ_obj(v) for v in list(x)[:max_list_elems]],
        }
    if isinstance(x, dict):
        keys = list(x.keys())
        return {
            "type": "dict",
            "keys": keys[:20],
            "num_keys": len(keys),
        }
    return {"type": type(x).__name__, "repr": repr(x)[:200]}


def main():
    ap = argparse.ArgumentParser(description="Inspect a VAE forward-trace .pt file structure.")
    ap.add_argument("trace_pt", help="Path to teletron_vae_trace.pt")
    ap.add_argument("--max-modules", type=int, default=10, help="Print at most N module records")
    ap.add_argument("--show-tensors", action="store_true", help="If set, print tensor stats when tensors are stored")
    args = ap.parse_args()

    path = Path(args.trace_pt)
    if not path.exists():
        raise SystemExit(f"file not found: {path}")

    obj = torch.load(path, map_location="cpu")
    print(f"Loaded: {path}")
    print(f"Top-level type: {type(obj)}")

    if not isinstance(obj, dict):
        print("Top-level is not a dict. Summary:")
        print(_summ_obj(obj))
        return

    print("\n=== Top-level keys ===")
    for k in obj.keys():
        print(f" - {k}")

    # 常见字段：name / created_at / extra / records / modules / trace 等
    # 我们尽量兼容不同实现
    extra = obj.get("extra", None) or obj.get("meta", None) or obj.get("metadata", None)
    if extra is not None:
        print("\n=== extra/meta ===")
        if isinstance(extra, dict):
            for k, v in extra.items():
                print(f"{k}: {_summ_obj(v)}")
        else:
            print(_summ_obj(extra))

    # 找记录体
    records = None
    for cand in ["records", "trace", "events", "module_records", "data"]:
        if cand in obj:
            records = obj[cand]
            records_key = cand
            break

    if records is None:
        print("\nNo obvious records key found. Showing top-level summaries:")
        for k, v in obj.items():
            print(f"\n[{k}] -> {_summ_obj(v)}")
        return

    print(f"\n=== Records key: '{records_key}' type={type(records)} ===")

    # 情况1：records 是 dict: module_name -> record
    if isinstance(records, dict):
        module_names = list(records.keys())
        print(f"Num modules recorded: {len(module_names)}")
        for i, name in enumerate(module_names[: args.max_modules]):
            rec = records[name]
            print(f"\n--- Module[{i}] {name} ---")
            print(f"record type={type(rec)}")
            if isinstance(rec, dict):
                print("record keys:", list(rec.keys()))
                for kk in list(rec.keys())[:20]:
                    vv = rec[kk]
                    if args.show_tensors and _is_tensor(vv):
                        print(f"  {kk}: tensor {_summ_tensor(vv)}")
                    else:
                        print(f"  {kk}: {_summ_obj(vv)}")
            else:
                print(_summ_obj(rec))

    # 情况2：records 是 list: 每次 hook 一个 event
    elif isinstance(records, list):
        print(f"Num events recorded: {len(records)}")
        for i, ev in enumerate(records[: args.max_modules]):
            print(f"\n--- Event[{i}] type={type(ev)} ---")
            if isinstance(ev, dict):
                # 常见字段：name / module / idx / input / output / stats ...
                print("event keys:", list(ev.keys()))
                for kk in list(ev.keys())[:30]:
                    vv = ev[kk]
                    if args.show_tensors and _is_tensor(vv):
                        print(f"  {kk}: tensor {_summ_tensor(vv)}")
                    else:
                        print(f"  {kk}: {_summ_obj(vv)}")
            else:
                print(_summ_obj(ev))
    else:
        print("records is neither dict nor list. Summary:")
        print(_summ_obj(records))


if __name__ == "__main__":
    main()
