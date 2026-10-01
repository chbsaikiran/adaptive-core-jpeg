#!/usr/bin/env python3
"""
analyze_overhead.py -- classifier overhead (feature extraction + inference)
as a share of total adaptive encode time, from a jcadaptivebench-static
phase CSV. Needs no power data, so it also runs on a machine with no power
sensor.

Per image, from the `adaptive` phase (which does exactly what
jcadaptive.c does per image, minus image decode and file output):

    feature_ms, inference_ms, encode_ms   mean per repeat
    overhead_share = (feature + inference) / (feature + inference + encode)

and, from the `always2`/`always4` phases, how long a plain fixed-core
encode of the same image takes -- the real question being whether
classifying first costs more time than simply encoding at the "wrong"
core count would have.

Standard library only; Python 3.6 compatible.
"""
import argparse
import csv
import sys


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def median(values):
    s = sorted(values)
    n = len(s)
    if n == 0:
        return float("nan")
    return s[n // 2] if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phases_csv", help="jcadaptivebench-static output")
    ap.add_argument("-o", "--output", help="per-image overhead CSV to write")
    args = ap.parse_args()

    images, order = {}, []
    with open(args.phases_csv, newline="") as f:
        for row in csv.DictReader(f):
            if row.get("phase") in (None, "idle"):
                continue
            if row["image"] not in images:
                images[row["image"]] = {}
                order.append(row["image"])
            images[row["image"]][row["phase"]] = row

    rows = []
    for name in order:
        ad = images[name].get("adaptive")
        if ad is None:
            continue
        reps = int(ad["reps"])
        feature = float(ad["feature_s"]) / reps
        inference = float(ad["inference_s"]) / reps
        encode = float(ad["encode_s"]) / reps
        classify = feature + inference
        fixed = {}
        for pol in ("always2", "always4"):
            r = images[name].get(pol)
            fixed[pol] = float(r["encode_s"]) / int(r["reps"]) if r else None
        mpx = int(ad["width"]) * int(ad["height"]) / 1e6
        rows.append({
            "image": name, "width": ad["width"], "height": ad["height"],
            "mpx": mpx, "label": int(ad["predicted_label"]),
            "threads": int(ad["threads"]), "feature": feature,
            "inference": inference, "encode": encode, "classify": classify,
            "share": classify / (classify + encode),
            "always2": fixed["always2"], "always4": fixed["always4"],
        })

    if not rows:
        sys.exit("%s: no adaptive-phase rows" % args.phases_csv)

    if args.output:
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["image", "width", "height", "predicted",
                        "adaptive_threads", "feature_ms", "inference_ms",
                        "encode_ms", "adaptive_total_ms", "overhead_share",
                        "classify_ms_per_mpx", "always2_encode_ms",
                        "always4_encode_ms"])
            for r in rows:
                w.writerow([
                    r["image"], r["width"], r["height"],
                    "simple" if r["label"] == 0 else "complex", r["threads"],
                    "%.4f" % (r["feature"] * 1000),
                    "%.4f" % (r["inference"] * 1000),
                    "%.4f" % (r["encode"] * 1000),
                    "%.4f" % ((r["classify"] + r["encode"]) * 1000),
                    "%.4f" % r["share"],
                    "%.4f" % (r["classify"] * 1000 / r["mpx"]),
                    "" if r["always2"] is None else "%.4f" % (r["always2"] * 1000),
                    "" if r["always4"] is None else "%.4f" % (r["always4"] * 1000),
                ])

    def report(title, subset):
        if not subset:
            return
        feature = sum(r["feature"] for r in subset)
        inference = sum(r["inference"] for r in subset)
        encode = sum(r["encode"] for r in subset)
        shares = [r["share"] for r in subset]
        n = len(subset)
        print("\n%s (%d images)" % (title, n))
        print("  mean per image: features %.3f ms, inference %.4f ms, "
              "encode %.3f ms" % (feature / n * 1000, inference / n * 1000,
                                  encode / n * 1000))
        print("  overhead share of adaptive time: overall %.1f%% "
              "(total classify / total time), per-image median %.1f%%, "
              "min %.1f%%, max %.1f%%" % (
                  100.0 * (feature + inference) / (feature + inference + encode),
                  100.0 * median(shares), 100.0 * min(shares),
                  100.0 * max(shares)))
        print("  feature extraction cost: %.3f ms per megapixel" % mean(
            [r["classify"] * 1000 / r["mpx"] for r in subset]))
        for pol in ("always2", "always4"):
            both = [r for r in subset if r[pol] is not None]
            if not both:
                continue
            adaptive_total = sum(r["classify"] + r["encode"] for r in both)
            fixed_total = sum(r[pol] for r in both)
            print("  adaptive total time vs %s encode: %.2fx (%.3f ms vs "
                  "%.3f ms per image)" % (
                      pol, adaptive_total / fixed_total,
                      adaptive_total / len(both) * 1000,
                      fixed_total / len(both) * 1000))

    report("All images", rows)
    report("Predicted simple -> 2 cores", [r for r in rows if r["label"] == 0])
    report("Predicted complex -> 4 cores", [r for r in rows if r["label"] == 1])
    if args.output:
        print("\nper-image table: %s" % args.output)


if __name__ == "__main__":
    main()
