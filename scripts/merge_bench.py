#!/usr/bin/env python3
"""
merge_bench.py -- join two jcparallelbench-static per-image CSVs (one run
at --threads 2, one at --threads 4, over the *same* dataset_dir) into a
single per-image dataset keyed by image filename:

    image,width,height,mean_2core_s,min_2core_s,mean_4core_s,min_4core_s,speedup_min

`speedup_min` = min_2core_s / min_4core_s is the column meant to drive
complexity labeling downstream (e.g. "complex" if 4 cores meaningfully beat
2 cores, "simple" otherwise) -- see scripts/bench_2vs4.sh, which calls this.

Images present in only one of the two input CSVs (e.g. one run's dataset_dir
had a file the other didn't, or an image failed to decode/encode on only
one run) are dropped with a warning rather than silently guessed at.
"""
import argparse
import csv
import sys


def load_csv(path):
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            image = row.get("image")
            if not image or image.startswith("#"):
                continue
            rows[image] = row
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv_2core", help="threads=2 CSV from run_bench.sh")
    ap.add_argument("csv_4core", help="threads=4 CSV from run_bench.sh")
    ap.add_argument("-o", "--output", required=True, help="merged dataset CSV to write")
    args = ap.parse_args()

    rows_2 = load_csv(args.csv_2core)
    rows_4 = load_csv(args.csv_4core)

    common = sorted(set(rows_2) & set(rows_4))
    for name in sorted(set(rows_2) - set(rows_4)):
        print(f"warning: {name}: in {args.csv_2core} but not {args.csv_4core}, dropping",
              file=sys.stderr)
    for name in sorted(set(rows_4) - set(rows_2)):
        print(f"warning: {name}: in {args.csv_4core} but not {args.csv_2core}, dropping",
              file=sys.stderr)

    fields = ["image", "width", "height", "mean_2core_s", "min_2core_s",
              "mean_4core_s", "min_4core_s", "speedup_min"]
    speedups = []
    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fields)
        for name in common:
            r2, r4 = rows_2[name], rows_4[name]
            mean2, min2 = float(r2["mean_time_s"]), float(r2["min_time_s"])
            mean4, min4 = float(r4["mean_time_s"]), float(r4["min_time_s"])
            speedup = (min2 / min4) if min4 > 0 else float("inf")
            speedups.append(speedup)
            w.writerow([name, r2["width"], r2["height"], f"{mean2:.6f}",
                        f"{min2:.6f}", f"{mean4:.6f}", f"{min4:.6f}",
                        f"{speedup:.4f}"])

    print(f"{len(common)} images merged -> {args.output}", file=sys.stderr)
    if speedups:
        print(f"speedup_min (2-core/4-core): min={min(speedups):.3f} "
              f"mean={sum(speedups) / len(speedups):.3f} max={max(speedups):.3f}",
              file=sys.stderr)
    else:
        print("warning: no images in common between the two runs -- "
              "dataset.csv has no rows", file=sys.stderr)


if __name__ == "__main__":
    main()
