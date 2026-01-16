import argparse
import glob
import os
from typing import Dict, List

import torch


def _collect_files(root: str) -> List[str]:
    pattern = os.path.join(root, "dpo_loss_compare_iter*_rank*.pt")
    return sorted(glob.glob(pattern))


def _format_bool(val):
    return "Y" if val else "N"


def _safe_get(mapping: Dict, key: str, default=""):
    return mapping.get(key, default)


def _flatten_compare(compare: Dict, prefix=""):
    rows = []
    for key, value in compare.items():
        name = f"{prefix}{key}" if prefix == "" else f"{prefix}.{key}"
        if isinstance(value, dict):
            rows.extend(_flatten_compare(value, name))
        else:
            rows.append((name, value))
    return rows


def _write_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            f.write(",".join(str(row.get(k, "")) for k in keys) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Parse Teletron DPO loss compare dumps.")
    parser.add_argument("--dir", required=True, help="Directory with dpo_loss_compare_iter*_rank*.pt")
    parser.add_argument("--out", default=None, help="Output CSV path")
    parser.add_argument("--only-core", action="store_true", help="Only keep dpo_loss/loss_chosen/loss_reject")
    args = parser.parse_args()

    files = _collect_files(args.dir)
    if not files:
        raise RuntimeError(f"No compare files found in {args.dir}")

    rows = []
    for path in files:
        payload = torch.load(path, map_location="cpu")
        meta = payload.get("meta", {})
        compare = payload.get("compare", {})
        row = {
            "iter": meta.get("iter", ""),
            "rank": meta.get("dp_rank", meta.get("rank", "")),
            "path": os.path.basename(path),
        }

        # top-level compare stats
        for loss_name, stats in compare.items():
            if not isinstance(stats, dict):
                continue
            if args.only_core and loss_name not in ("dpo_loss", "loss_chosen", "loss_reject"):
                continue
            row[f"{loss_name}.allclose"] = _format_bool(stats.get("allclose"))
            row[f"{loss_name}.max_abs"] = stats.get("max_abs", "")
            row[f"{loss_name}.mean_abs"] = stats.get("mean_abs", "")

        # include saved/current means if present
        for key, val in payload.items():
            if key.startswith("current_mean/") or key.startswith("saved_mean/"):
                if args.only_core:
                    if not key.endswith("dpo_loss") and not key.endswith("loss_chosen") and not key.endswith("loss_reject"):
                        continue
                row[key] = val

        rows.append(row)

    out_path = args.out or os.path.join(args.dir, "dpo_loss_compare_summary.csv")
    _write_csv(out_path, rows)
    print(f"[OK] wrote {out_path}")


if __name__ == "__main__":
    main()
