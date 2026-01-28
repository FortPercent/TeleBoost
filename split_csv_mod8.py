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


def split_csv(csv_path: str, out_dir: str, num_parts: int = 8, has_header: bool = True) -> int:
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(os.path.basename(csv_path))[0]
    out_paths = [os.path.join(out_dir, f"{base}.part{i}.csv") for i in range(num_parts)]

    writers = []
    out_files = []
    try:
        # Open output files
        for p in out_paths:
            fo = open(p, "w", newline="", encoding="utf-8")
            out_files.append(fo)
            writers.append(csv.writer(fo))

        # Read input and write
        with open(csv_path, "r", newline="", encoding="utf-8") as fi:
            reader = csv.reader(fi)

            header = None
            if has_header:
                header = next(reader, None)
                if header is not None:
                    for w in writers:
                        w.writerow(header)

            data_idx = 0  # 0-based index among data rows
            for row in reader:
                part_id = data_idx % num_parts
                writers[part_id].writerow(row)
                data_idx += 1

        return data_idx  # total data rows processed
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

    processed = split_csv(args.input, args.out_dir, num_parts=args.parts, has_header=has_header)
    print(f"[INFO] Split done. Processed data rows: {processed}")
    print(f"[INFO] Output files are in: {args.out_dir}")


if __name__ == "__main__":
    main()

