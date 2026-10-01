#!/usr/bin/env python3
"""
power_logger.py -- sample the board's power rails to a CSV until stopped
(SIGINT/SIGTERM), timestamping every sample with CLOCK_MONOTONIC so the
samples line up with the phase timestamps jcadaptivebench-static prints.

Output CSV: `t_s,<rail>_mw[,<rail>_mw...]`, the board's main input rail
first. All power values are milliwatts.

Power sources, tried in this order unless --source picks one:

  ina3221x   Original Jetson Nano / JetPack 4 (L4T r32). The on-board
             INA3221 is exposed under
             /sys/bus/i2c/drivers/ina3221x/*/iio:device*/ as
             rail_name_N + in_power_N_input (mW). Main rail: POM_5V_IN.
  hwmon      JetPack 5+ boards (e.g. Orin Nano). INA3221 exposed as hwmon:
             inN_label + inN_input (mV) + currN_input (mA). Main rail:
             VDD_IN.
  tegrastats Fallback: runs `tegrastats --interval MS` and parses the
             `POM_5V_IN cur/avg` or `VDD_IN cur mW/avg mW` field. Coarser
             (tegrastats' own interval floor) and timestamped on line
             arrival, so prefer the sysfs sources when they exist.
  file       --file PATH: any file holding one number, scaled to mW by
             --file-scale (for other boards or an external meter).

Standard library only, and Python 3.6 compatible (the Nano's JetPack 4
system python3), so nothing needs installing on the board.
"""
import argparse
import glob
import os
import re
import signal
import subprocess
import sys
import time

MAIN_RAILS = ("POM_5V_IN", "VDD_IN")

_stop = False


def _on_signal(signum, frame):
    global _stop
    _stop = True


def now():
    return time.clock_gettime(time.CLOCK_MONOTONIC)


def read_text(path):
    with open(path) as f:
        return f.read().strip()


def main_rail_first(rails):
    """Stable-sorts (name, reader) pairs so a known main input rail leads."""
    return sorted(rails, key=lambda r: 0 if r[0] in MAIN_RAILS else 1)


def find_ina3221x():
    rails = []
    for dev in sorted(glob.glob("/sys/bus/i2c/drivers/ina3221x/*/iio:device*")):
        for name_path in sorted(glob.glob(os.path.join(dev, "rail_name_*"))):
            idx = name_path.rsplit("_", 1)[1]
            power_path = os.path.join(dev, "in_power%s_input" % idx)
            if not os.path.exists(power_path):
                continue
            rails.append((read_text(name_path),
                          lambda p=power_path: float(read_text(p))))
    return main_rail_first(rails)


def find_hwmon():
    rails = []
    for dev in sorted(glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*")):
        for label_path in sorted(glob.glob(os.path.join(dev, "in*_label"))):
            idx = re.search(r"in(\d+)_label$", label_path).group(1)
            volt_path = os.path.join(dev, "in%s_input" % idx)
            curr_path = os.path.join(dev, "curr%s_input" % idx)
            if not (os.path.exists(volt_path) and os.path.exists(curr_path)):
                continue
            # mV * mA / 1000 = mW
            rails.append((read_text(label_path),
                          lambda v=volt_path, c=curr_path:
                          float(read_text(v)) * float(read_text(c)) / 1000.0))
    return main_rail_first(rails)


def column_name(rail):
    return re.sub(r"[^a-z0-9]+", "_", rail.lower()).strip("_") + "_mw"


def log_polled(rails, interval_s, out):
    out.write("t_s," + ",".join(column_name(name) for name, _ in rails) + "\n")
    count = 0
    next_t = now()
    while not _stop:
        t = now()
        try:
            values = [reader() for _, reader in rails]
        except (OSError, ValueError) as e:
            print("power_logger: read failed: %s" % e, file=sys.stderr)
            return count
        out.write("%.6f,%s\n" % (t, ",".join("%.1f" % v for v in values)))
        count += 1
        next_t += interval_s
        delay = next_t - now()
        if delay > 0:
            time.sleep(delay)
        else:
            next_t = now()  # fell behind; don't try to catch up in a burst
    return count


TEGRASTATS_RE = re.compile(r"\b(POM_5V_IN|VDD_IN) (\d+)(?:mW)?/\d+")


def parse_tegrastats_line(line):
    """Returns (rail_name, current_mw) from one tegrastats line, or None."""
    m = TEGRASTATS_RE.search(line)
    if not m:
        return None
    return m.group(1), float(m.group(2))


def log_tegrastats(interval_s, out):
    interval_ms = max(1, int(round(interval_s * 1000)))
    proc = subprocess.Popen(["tegrastats", "--interval", str(interval_ms)],
                            stdout=subprocess.PIPE, universal_newlines=True)
    count = 0
    try:
        for line in proc.stdout:
            t = now()
            parsed = parse_tegrastats_line(line)
            if parsed is None:
                continue
            if count == 0:
                out.write("t_s,%s\n" % column_name(parsed[0]))
            out.write("%.6f,%.1f\n" % (t, parsed[1]))
            count += 1
            if _stop:
                break
    finally:
        proc.terminate()
    return count


def have_tegrastats():
    return any(os.access(os.path.join(d, "tegrastats"), os.X_OK)
               for d in os.environ.get("PATH", "").split(os.pathsep) if d)


def detect(source, file_path, file_scale):
    """Returns (source_name, rails) -- rails is None for tegrastats."""
    if file_path:
        return "file", [(os.path.basename(file_path),
                         lambda: float(read_text(file_path)) * file_scale)]
    if source in ("auto", "ina3221x"):
        rails = find_ina3221x()
        if rails:
            return "ina3221x", rails
    if source in ("auto", "hwmon"):
        rails = find_hwmon()
        if rails:
            return "hwmon", rails
    if source in ("auto", "tegrastats") and have_tegrastats():
        return "tegrastats", None
    return None, None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--output", help="CSV to write (default: stdout)")
    ap.add_argument("--interval-ms", type=float, default=10.0,
                    help="sampling interval (default 10 ms)")
    ap.add_argument("--source", default="auto",
                    choices=["auto", "ina3221x", "hwmon", "tegrastats"])
    ap.add_argument("--file", help="read power from this file instead")
    ap.add_argument("--file-scale", type=float, default=1.0,
                    help="multiply --file's value by this to get mW "
                         "(e.g. 0.001 for a file in microwatts)")
    ap.add_argument("--probe", action="store_true",
                    help="print the detected source and one reading, then exit "
                         "(exit status 1 if no power source was found)")
    args = ap.parse_args()

    source, rails = detect(args.source, args.file, args.file_scale)
    if source is None:
        print("power_logger: no power source found (not a Jetson, or the "
              "INA3221 sysfs nodes are missing) -- use --file, or see "
              "scripts/measure/README.md", file=sys.stderr)
        return 1

    if args.probe:
        if rails is None:
            print("source=tegrastats")
        else:
            print("source=%s" % source)
            for name, reader in rails:
                print("  %s = %.1f mW" % (name, reader()))
        return 0

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    out = open(args.output, "w") if args.output else sys.stdout
    try:
        if rails is None:
            count = log_tegrastats(args.interval_ms / 1000.0, out)
        else:
            count = log_polled(rails, args.interval_ms / 1000.0, out)
    finally:
        out.flush()
        if out is not sys.stdout:
            out.close()

    print("power_logger: %d samples from %s" % (count, source), file=sys.stderr)
    return 0 if count > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
