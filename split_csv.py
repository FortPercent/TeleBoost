#!/usr/bin/env python3
# split_csv_mod8.py
import csv
import os
import argparse


def count_rows(csv_path: str, has_header: bool = True) -> int:
    """Count data rows (excluding header if has_header=True)."""
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        if has_header:
            next(reader, None)  # skip header
        return sum(1 for _ in reader)


def split_csv(csv_path: str, out_dir: str, num_parts: int = 8, has_header: bool = True) -> tuple[int, int]:
    """Round-robin split, truncated to a multiple of num_parts so every part has equal row count.

    Returns (written_rows, dropped_rows). Equal row counts across parts prevents DP ranks
    from getting out of sync on collective ops (TP/CP all-reduce hang).
    """
    os.makedirs(out_dir, exist_ok=True)

    total_rows = count_rows(csv_path, has_header=has_header)
    keep_rows = (total_rows // num_parts) * num_parts
    dropped = total_rows - keep_rows

    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_paths = [os.path.join(out_dir, f"{base}.part{i}.csv") for i in range(num_parts)]

    writers = []
    out_files = []
    try:
        for p in out_paths:
            fo = open(p, "w", newline="", encoding="utf-8")
            out_files.append(fo)
            writers.append(csv.writer(fo))

        with open(csv_path, "r", newline="", encoding="utf-8") as fi:
            reader = csv.reader(fi)

            if has_header:
                header = next(reader, None)
                if header is not None:
                    for w in writers:
                        w.writerow(header)

            for data_idx, row in enumerate(reader):
                if data_idx >= keep_rows:
                    break
                writers[data_idx % num_parts].writerow(row)

        return keep_rows, dropped
    finally:
        for fo in out_files:
            try:
                fo.close()
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser(description="Split a CSV into N parts by taking every N-th row.")
    ap.add_argument("--input", "-i", required=True, help="Input CSV path")
    ap.add_argument("--out_dir", "-o", required=True, help="Output directory")
    ap.add_argument("--parts", "-p", type=int, default=8, help="Number of parts (default: 8)")
    ap.add_argument("--no_header", action="store_true", help="Treat CSV as having NO header row")
    args = ap.parse_args()

    has_header = not args.no_header

    total = count_rows(args.input, has_header=has_header)
    print(f"[INFO] Total data rows: {total}")

    written, dropped = split_csv(args.input, args.out_dir, num_parts=args.parts, has_header=has_header)
    per_part = written // args.parts
    print(f"[INFO] Split done. Written: {written} rows ({per_part}/part), dropped: {dropped} for divisibility.")
    print(f"[INFO] Output files are in: {args.out_dir}")


if __name__ == "__main__":
    main()

