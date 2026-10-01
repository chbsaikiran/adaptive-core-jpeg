#!/usr/bin/env python3
"""
analyze_energy.py -- energy per image for the always-2-core, always-4-core
and adaptive policies, from a jcadaptivebench-static phase CSV plus a
power_logger.py power CSV recorded over the same run.

For each phase row, the power samples falling in [t_start_s, t_end_s] are
integrated (trapezoid rule, with the window's two edges linearly
interpolated between the neighbouring samples) to get the phase's total
energy, which is divided by the phase's repeat count:

    gross_mj = E_phase / reps
    net_mj   = (E_phase - P_idle * elapsed) / reps

`gross` is everything the board drew while encoding one image. `net`
subtracts the board's idle draw (mean power over the run's "idle" phases),
leaving the energy attributable to the work itself. Which one to headline
depends on the claim: `net` isolates the encoder's own cost, while `gross`
is what a battery actually sees -- and under `gross`, finishing sooner
saves idle energy too, which tends to favour 4 cores. Report both.

mW * s = mJ, so all energies are millijoules.

Standard library only; Python 3.6 compatible.
"""
import argparse
import bisect
import csv
import sys

POLICIES = ("always2", "always4", "adaptive")


def load_phases(path):
    with open(path, newline="") as f:
        return [row for row in csv.DictReader(f) if row.get("phase")]


def load_power(path, rail):
    """Returns (times, milliwatts, column_name) for the chosen rail."""
    times, power = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header or len(header) < 2:
            sys.exit("%s: empty or malformed power log" % path)
        if rail is None:
            col = 1
        else:
            matches = [i for i, name in enumerate(header)
                       if i > 0 and rail.lower() in name.lower()]
            if not matches:
                sys.exit("%s: no rail matching %r (columns: %s)"
                         % (path, rail, ", ".join(header[1:])))
            col = matches[0]
        for row in reader:
            try:
                t, p = float(row[0]), float(row[col])
            except (ValueError, IndexError):
                continue
            times.append(t)
            power.append(p)
    return times, power, header[col]


def power_at(times, power, t):
    """Power at time t, linearly interpolated; clamped outside the log."""
    i = bisect.bisect_left(times, t)
    if i == 0:
        return power[0]
    if i == len(times):
        return power[-1]
    t0, t1 = times[i - 1], times[i]
    if t1 == t0:
        return power[i]
    return power[i - 1] + (power[i] - power[i - 1]) * (t - t0) / (t1 - t0)


def integrate(times, power, t_start, t_end):
    """Returns (energy_mj, samples_inside_window) over [t_start, t_end]."""
    lo = bisect.bisect_right(times, t_start)
    hi = bisect.bisect_left(times, t_end)
    ts = [t_start] + times[lo:hi] + [t_end]
    ps = ([power_at(times, power, t_start)] + power[lo:hi] +
          [power_at(times, power, t_end)])
    energy = 0.0
    for k in range(1, len(ts)):
        energy += 0.5 * (ps[k - 1] + ps[k]) * (ts[k] - ts[k - 1])
    return energy, hi - lo


def mean(values):
    return sum(values) / len(values) if values else float("nan")


def pct_change(new, base):
    return 100.0 * (new - base) / base if base else float("nan")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phases_csv", help="jcadaptivebench-static output")
    ap.add_argument("power_csv", help="power_logger.py output")
    ap.add_argument("-o", "--output", help="per-image energy CSV to write")
    ap.add_argument("--rail", help="power column to use, by (partial) name; "
                                   "default: the first one (main input rail)")
    ap.add_argument("--min-samples", type=int, default=20,
                    help="warn about phases with fewer power samples than "
                         "this (default 20)")
    args = ap.parse_args()

    phases = load_phases(args.phases_csv)
    times, power, rail = load_power(args.power_csv, args.rail)
    if len(times) < 2:
        sys.exit("%s: fewer than 2 power samples" % args.power_csv)

    covered = [p for p in phases
               if float(p["t_start_s"]) >= times[0]
               and float(p["t_end_s"]) <= times[-1]]
    if len(covered) < len(phases):
        print("warning: %d of %d phases fall outside the power log's time "
              "range and are dropped -- was the logger started before the "
              "benchmark and stopped after it?"
              % (len(phases) - len(covered), len(phases)), file=sys.stderr)

    idle_energy = idle_time = 0.0
    for p in covered:
        if p["phase"] == "idle":
            e, _ = integrate(times, power, float(p["t_start_s"]),
                             float(p["t_end_s"]))
            idle_energy += e
            idle_time += float(p["elapsed_s"])
    if idle_time <= 0:
        sys.exit("no idle phase inside the power log -- can't compute the "
                 "idle baseline")
    idle_mw = idle_energy / idle_time

    # image -> {policy: {...}}, plus the classify phase's prediction
    images, order, sparse = {}, [], 0
    for p in covered:
        if p["phase"] == "idle":
            continue
        t0, t1 = float(p["t_start_s"]), float(p["t_end_s"])
        elapsed, reps = float(p["elapsed_s"]), int(p["reps"])
        energy, n = integrate(times, power, t0, t1)
        if n < args.min_samples:
            sparse += 1
        if p["image"] not in images:
            images[p["image"]] = {}
            order.append(p["image"])
        images[p["image"]][p["phase"]] = {
            "threads": int(p["threads"]),
            "time_s": elapsed / reps,
            "gross_mj": energy / reps,
            "net_mj": (energy - idle_mw * elapsed) / reps,
            "mean_mw": energy / elapsed,
            "label": int(p["predicted_label"]),
            "width": p["width"], "height": p["height"],
        }
    if sparse:
        print("warning: %d phases had fewer than %d power samples -- raise "
              "--min-seconds in the benchmark or lower the logger's "
              "--interval-ms" % (sparse, args.min_samples), file=sys.stderr)

    complete = [name for name in order
                if all(pol in images[name] for pol in POLICIES)]
    if not complete:
        sys.exit("no image has all of always2/always4/adaptive measured")

    if args.output:
        fields = ["image", "width", "height", "predicted", "adaptive_threads"]
        for pol in POLICIES:
            fields += [pol + "_time_s", pol + "_gross_mj", pol + "_net_mj",
                       pol + "_mean_mw"]
        fields += ["classify_time_s", "classify_net_mj"]
        with open(args.output, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(fields)
            for name in complete:
                im = images[name]
                ad = im["adaptive"]
                row = [name, ad["width"], ad["height"],
                       "simple" if ad["label"] == 0 else "complex",
                       ad["threads"]]
                for pol in POLICIES:
                    row += ["%.6f" % im[pol]["time_s"],
                            "%.3f" % im[pol]["gross_mj"],
                            "%.3f" % im[pol]["net_mj"],
                            "%.1f" % im[pol]["mean_mw"]]
                cl = im.get("classify")
                row += ["%.6f" % cl["time_s"] if cl else "",
                        "%.3f" % cl["net_mj"] if cl else ""]
                w.writerow(row)

    def report(title, names):
        if not names:
            return
        print("\n%s (%d images)" % (title, len(names)))
        print("  %-9s %12s %13s %11s %13s" % (
            "policy", "time/img ms", "gross mJ/img", "net mJ/img", "mean power mW"))
        stats = {}
        for pol in POLICIES:
            stats[pol] = {k: mean([images[n][pol][k] for n in names])
                          for k in ("time_s", "gross_mj", "net_mj", "mean_mw")}
            s = stats[pol]
            print("  %-9s %12.2f %13.2f %11.2f %13.0f" % (
                pol, s["time_s"] * 1000, s["gross_mj"], s["net_mj"],
                s["mean_mw"]))
        for base in ("always4", "always2"):
            print("  adaptive vs %s: time %+.1f%%, gross energy %+.1f%%, "
                  "net energy %+.1f%%" % (
                      base,
                      pct_change(stats["adaptive"]["time_s"],
                                 stats[base]["time_s"]),
                      pct_change(stats["adaptive"]["gross_mj"],
                                 stats[base]["gross_mj"]),
                      pct_change(stats["adaptive"]["net_mj"],
                                 stats[base]["net_mj"])))

    print("power rail: %s   samples: %d   idle power: %.0f mW (over %.1f s)"
          % (rail, len(times), idle_mw, idle_time))
    report("All images", complete)
    report("Predicted simple -> adaptive used 2 cores",
           [n for n in complete if images[n]["adaptive"]["label"] == 0])
    report("Predicted complex -> adaptive used 4 cores",
           [n for n in complete if images[n]["adaptive"]["label"] == 1])

    classify = [images[n]["classify"]["net_mj"] for n in complete
                if "classify" in images[n]]
    if classify:
        print("\nclassifier alone (features + inference): mean net energy "
              "%.2f mJ/img" % mean(classify))
    if args.output:
        print("\nper-image table: %s" % args.output)


if __name__ == "__main__":
    main()
