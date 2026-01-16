import argparse
import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt
    HAS_MPL = True
except Exception:
    HAS_MPL = False


FILENAME_RE = re.compile(r"dpo_inputs_iter(\d+)_rank(\d+)\.pt$")


def _parse_list(arg: str) -> List[int]:
    if not arg:
        return []
    return [int(x) for x in arg.split(",") if x.strip()]


def _flatten_tensors(obj, prefix="", out=None):
    if out is None:
        out = {}
    if torch.is_tensor(obj):
        out[prefix] = obj
        return out
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "meta":
                continue
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            _flatten_tensors(v, new_prefix, out)
        return out
    if isinstance(obj, (list, tuple)):
        for idx, v in enumerate(obj):
            new_prefix = f"{prefix}[{idx}]"
            _flatten_tensors(v, new_prefix, out)
        return out
    return out


def _select_samples(a: torch.Tensor, b: torch.Tensor, sample_size: int):
    if a.shape != b.shape:
        return None, None, 0
    a_flat = a.flatten()
    b_flat = b.flatten()
    numel = a_flat.numel()
    if sample_size and numel > sample_size:
        idx = torch.randperm(numel)[:sample_size]
        return a_flat[idx], b_flat[idx], sample_size
    return a_flat, b_flat, numel


def _compute_stats(a: torch.Tensor, b: torch.Tensor, sample_size: int, rtol: float, atol: float):
    if a.shape != b.shape:
        return {
            "shape_mismatch": True,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }
    a_use, b_use, sampled = _select_samples(a, b, sample_size)
    numel = a.numel()
    if a_use is None:
        return {
            "shape_mismatch": True,
            "shape_a": list(a.shape),
            "shape_b": list(b.shape),
        }

    diff = (a_use - b_use).abs()
    mean_abs = diff.mean().item()
    max_abs = diff.max().item()
    l2_diff = torch.norm(a_use - b_use).item()
    l2_ref = torch.norm(a_use).item()
    mean_ref = a_use.abs().mean().item()
    eps = 1e-12

    allclose = torch.allclose(a_use, b_use, rtol=rtol, atol=atol)
    return {
        "shape_mismatch": False,
        "numel": numel,
        "sampled": sampled,
        "mean_abs": mean_abs,
        "max_abs": max_abs,
        "mean_rel": mean_abs / (mean_ref + eps),
        "l2_rel": l2_diff / (l2_ref + eps),
        "allclose": bool(allclose),
    }


def _collect_files(inputs_dir: str) -> Dict[int, Dict[int, str]]:
    files = glob.glob(os.path.join(inputs_dir, "dpo_inputs_iter*_rank*.pt"))
    by_iter: Dict[int, Dict[int, str]] = {}
    for path in files:
        name = os.path.basename(path)
        match = FILENAME_RE.match(name)
        if not match:
            continue
        it = int(match.group(1))
        rank = int(match.group(2))
        by_iter.setdefault(it, {})[rank] = path
    return by_iter


def _save_heatmap(matrix: np.ndarray, row_labels: List[str], col_labels: List[str], title: str, out_path: str):
    fig, ax = plt.subplots(figsize=(1 + 0.4 * len(col_labels), 1 + 0.25 * len(row_labels)))
    im = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_yticks(np.arange(len(row_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right")
    ax.set_yticklabels(row_labels)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compare saved DPO inputs across dp ranks.")
    parser.add_argument("--inputs-dir", required=True, help="Directory with dpo_inputs_iter*_rank*.pt")
    parser.add_argument("--iters", default="", help="Comma-separated iteration list (default: all found)")
    parser.add_argument("--ranks", default="", help="Comma-separated dp ranks (default: all found)")
    parser.add_argument("--ref-rank", type=int, default=None, help="Reference dp rank (default: min rank)")
    parser.add_argument("--sample-size", type=int, default=0, help="Sample elements per tensor for stats")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: <inputs-dir>/analysis)")
    parser.add_argument("--metric", default="max_abs", choices=["max_abs", "mean_abs", "mean_rel", "l2_rel"])
    parser.add_argument("--no-heatmap", action="store_true")
    parser.add_argument("--focus", default="context,chosen.,rejected.", help="Comma-separated prefixes to compare")
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=1e-8)
    args = parser.parse_args()

    by_iter = _collect_files(args.inputs_dir)
    if not by_iter:
        raise RuntimeError(f"No matching files found in {args.inputs_dir}")

    iters = _parse_list(args.iters) or sorted(by_iter.keys())
    output_dir = args.output_dir or os.path.join(args.inputs_dir, "analysis")
    os.makedirs(output_dir, exist_ok=True)

    for it in iters:
        if it not in by_iter:
            print(f"[WARN] iteration {it} not found, skipping")
            continue

        available_ranks = sorted(by_iter[it].keys())
        ranks = _parse_list(args.ranks) or available_ranks
        ranks = [r for r in ranks if r in by_iter[it]]
        if len(ranks) < 2:
            print(f"[WARN] iteration {it} has <2 ranks to compare, skipping")
            continue

        ref_rank = args.ref_rank if args.ref_rank is not None else min(ranks)
        if ref_rank not in by_iter[it]:
            raise RuntimeError(f"Reference rank {ref_rank} missing for iter {it}")

        ref_payload = torch.load(by_iter[it][ref_rank], weights_only=False, map_location="cpu")
        ref_tensors = _flatten_tensors(ref_payload)

        focus_prefixes = [p.strip() for p in args.focus.split(",") if p.strip()]
        tensor_keys = [k for k in sorted(ref_tensors.keys()) if any(k.startswith(p) for p in focus_prefixes)]
        if not tensor_keys:
            print(f"[WARN] iteration {it} has no tensors matching focus={args.focus}")
            continue

        rows = []
        heatmap_values = np.full((len(tensor_keys), len(ranks)), np.nan, dtype=np.float64)
        summary_rows = []

        for col_idx, rank in enumerate(ranks):
            if rank == ref_rank:
                continue
            payload = torch.load(by_iter[it][rank], weights_only=False, map_location="cpu")
            tensors = _flatten_tensors(payload)
            for row_idx, key in enumerate(tensor_keys):
                if key not in tensors:
                    rows.append({
                        "iter": it,
                        "rank": rank,
                        "ref_rank": ref_rank,
                        "tensor": key,
                        "dtype": str(ref_tensors[key].dtype),
                        "shape": list(ref_tensors[key].shape),
                        "missing": True,
                    })
                    continue
                stats = _compute_stats(ref_tensors[key], tensors[key], args.sample_size, args.rtol, args.atol)
                row = {
                    "iter": it,
                    "rank": rank,
                    "ref_rank": ref_rank,
                    "tensor": key,
                    "dtype": str(ref_tensors[key].dtype),
                    "shape": list(ref_tensors[key].shape),
                    "missing": False,
                }
                row.update(stats)
                rows.append(row)
                if not stats.get("shape_mismatch"):
                    heatmap_values[row_idx, col_idx] = stats.get(args.metric)

        for row_idx, key in enumerate(tensor_keys):
            per_rank = [r for r in rows if r.get("tensor") == key and not r.get("missing")]
            if not per_rank:
                continue
            allclose_flags = [r.get("allclose") for r in per_rank if r.get("shape_mismatch") is False]
            equal_all = all(allclose_flags) if allclose_flags else False
            max_abs = max((r.get("max_abs", float("nan")) for r in per_rank), default=float("nan"))
            mean_abs = max((r.get("mean_abs", float("nan")) for r in per_rank), default=float("nan"))
            max_rank = None
            max_val = -1.0
            for r in per_rank:
                val = r.get("max_abs", -1.0)
                if val is not None and val > max_val:
                    max_val = val
                    max_rank = r.get("rank")
            summary_rows.append({
                "iter": it,
                "tensor": key,
                "ref_rank": ref_rank,
                "ranks_compared": ",".join(str(r) for r in ranks if r != ref_rank),
                "equal_all": equal_all,
                "max_abs": max_abs,
                "mean_abs": mean_abs,
                "max_abs_rank": max_rank,
            })

        csv_path = os.path.join(output_dir, f"dpo_inputs_diff_iter{it}.csv")
        _write_csv(csv_path, rows)
        print(f"[OK] wrote {csv_path}")

        summary_path = os.path.join(output_dir, f"dpo_inputs_summary_iter{it}.csv")
        _write_csv(summary_path, summary_rows)
        print(f"[OK] wrote {summary_path}")

        diff_keys = [r["tensor"] for r in summary_rows if not r.get("equal_all")]
        if diff_keys:
            print(f"[WARN] iter {it} tensors differ across ranks: {len(diff_keys)}")

        if HAS_MPL and not args.no_heatmap:
            col_labels = [f"rank{r}" for r in ranks]
            row_labels = tensor_keys
            heatmap_path = os.path.join(output_dir, f"dpo_inputs_{args.metric}_iter{it}.png")
            _save_heatmap(
                heatmap_values,
                row_labels,
                col_labels,
                f"Iter {it} ({args.metric}) vs ref rank {ref_rank}",
                heatmap_path,
            )
            print(f"[OK] wrote {heatmap_path}")
        elif not HAS_MPL:
            print("[WARN] matplotlib not available, heatmap skipped")


def _write_csv(path: str, rows: List[Dict]):
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for row in rows:
            values = [str(row.get(k, "")) for k in keys]
            f.write(",".join(values) + "\n")


if __name__ == "__main__":
    main()
