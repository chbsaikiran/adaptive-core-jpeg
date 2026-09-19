"""Derive simple/complex labels from the jcparallelbench timing CSVs.

Raw mean_2core_s is dominated by resolution -- a big flat image can take
longer to encode than a small busy one purely because it has more pixels.
We normalize by megapixels first so the label reflects content difficulty,
not image size, then threshold on a percentile of that normalized value.

speedup_min (min_2core_s / min_4core_s) was considered as the label basis
instead, but in the benchmark data it never drops below ~1.66, so an
">1 = complex" rule (as sketched in Steps_to_build_and_benchmark_*.md)
would label every image complex.
"""

import argparse
import csv
from pathlib import Path


def load_rows(csv_path: Path) -> list[dict]:
    with csv_path.open(newline="") as f:
        return list(csv.DictReader(f))


def label_rows(rows: list[dict], simple_fraction: float) -> list[dict]:
    for row in rows:
        megapixels = (int(row["width"]) * int(row["height"])) / 1_000_000
        row["time_per_mpx"] = float(row["mean_2core_s"]) / megapixels

    sorted_times = sorted(float(row["time_per_mpx"]) for row in rows)
    cutoff_index = round(len(sorted_times) * simple_fraction)
    cutoff_index = min(max(cutoff_index, 0), len(sorted_times) - 1)
    threshold = sorted_times[cutoff_index]

    for row in rows:
        row["label"] = 0 if float(row["time_per_mpx"]) <= threshold else 1

    return rows


def write_rows(rows: list[dict], out_path: Path) -> None:
    fieldnames = [*rows[0].keys()]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", type=Path, default=Path("../../bench_results"))
    parser.add_argument(
        "--simple-fraction",
        type=float,
        default=0.17,
        help="Fraction of images (by lowest time-per-megapixel) labeled simple (0).",
    )
    args = parser.parse_args()

    for split in ("train", "val"):
        in_path = args.bench_dir / f"dataset_{split}.csv"
        out_path = args.bench_dir / f"labels_{split}.csv"
        rows = load_rows(in_path)
        rows = label_rows(rows, args.simple_fraction)
        write_rows(rows, out_path)

        simple_count = sum(1 for row in rows if row["label"] == 0)
        complex_count = len(rows) - simple_count
        print(f"{split}: {simple_count} simple, {complex_count} complex -> {out_path}")


if __name__ == "__main__":
    main()
