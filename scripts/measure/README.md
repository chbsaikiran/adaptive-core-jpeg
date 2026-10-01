# Energy and classifier-overhead measurements

Two measurements for the adaptive-core encoder, both over a directory of
real images:

1. **Energy per image** for three core-allocation policies — always 2
   cores, always 4 cores, and adaptive (classifier picks 2 or 4). Needs a
   Jetson's on-board power sensor.
2. **Classifier overhead** — time for feature extraction plus inference, as
   a share of total adaptive encode time. Needs no power sensor, so it runs
   anywhere (Mac included).

## Quick start

On the Jetson Nano, after building per
[Steps_to_build_and_run_jetson_nano.md](../../Steps_to_build_and_run_jetson_nano.md)
(`cmake -DWITH_OPENMP=1 -B build`):

```bash
# Put the board in a fixed, known power state first (see "Before you
# measure" below).
sudo nvpmodel -m 0
sudo jetson_clocks

./scripts/measure/run_measure.sh testimages/DIV2K_valid_HR
```

Arguments: `run_measure.sh [dataset_dir] [min_seconds] [limit]`

- `dataset_dir` — default `testimages/DIV2K_valid_HR` (the 100 images the
  classifier was *not* trained on; copy them to the Jetson first, they are
  gitignored).
- `min_seconds` — how long each phase is stretched, default `2`.
- `limit` — only the first N images, e.g. `./scripts/measure/run_measure.sh
  testimages/DIV2K_valid_HR 2 5` for a quick trial.

A full default run is 100 images x 4 phases x 2 s, roughly 15 minutes plus
image decode time.

Output, all in `bench_results/measure/` (gitignored):

| file | contents |
|---|---|
| `phases.csv` | raw benchmark output: one row per (image, phase) |
| `power.csv` | raw power samples (mW) |
| `energy_summary.txt`, `energy_per_image.csv` | energy results |
| `overhead_summary.txt`, `overhead_per_image.csv` | overhead results |
| `run_info.txt` | date, host, power mode, OpenMP wait policy, power source |

## How it works

| piece | role |
|---|---|
| [`src/jcadaptivebench.c`](../../src/jcadaptivebench.c) (`jcadaptivebench-static`) | decodes each image once, then runs each policy as a timestamped *phase*, repeating the work until the phase has lasted `min_seconds` |
| `power_logger.py` | samples the power rail every 10 ms, timestamped on the same clock (`CLOCK_MONOTONIC`) |
| `analyze_energy.py` | integrates power over each phase's time window, divides by the repeat count |
| `analyze_overhead.py` | computes overhead from the per-stage timings in `phases.csv` |
| `run_measure.sh` | runs all of the above in order |

Phases per image: `classify` (features + inference only), `always2`,
`always4`, `adaptive` (features + inference + encode at the predicted core
count — what `jcadaptive-static` does per image, minus image decode and
file output). The run starts and ends with an `idle` phase, used as the
baseline power.

A single encode takes tens of milliseconds, and the power sensor only
refreshes every few milliseconds, so one encode cannot be measured
directly. Each phase therefore repeats the same work for `min_seconds` and
the energy is divided by the number of repeats.

The three encode phases run in a rotating order from image to image, so no
policy always runs right after the same neighbour (thermal and
frequency-scaling carry-over).

## Reading the energy results

`analyze_energy.py` reports two energies per image, in millijoules:

- **gross** — everything the board drew while encoding the image.
- **net** — gross minus the board's idle draw over the same time, i.e. the
  energy attributable to the encode itself.

They can disagree about which policy wins. Under *gross*, a policy that
finishes sooner also stops paying the board's idle power sooner, which
favours 4 cores. Under *net*, only the extra power drawn by the work
counts. Report both and say which one your claim rests on: *gross* is what
a battery sees if the board would otherwise sit idle or sleep; *net* is the
fairer number if the board is busy with other work regardless.

The summary is printed for all images and separately for images predicted
simple (adaptive used 2 cores) and complex (adaptive used 4 cores). Note
that for a predicted-complex image, adaptive does the same encode as
always-4 plus the classifier's work, so it can only come out worse there;
any saving has to come from the predicted-simple images.

## Before you measure

These all change the numbers, so fix them and record them (`run_info.txt`
captures the power mode and wait policy):

- **Power mode.** `sudo nvpmodel -m 0` (MAXN, all 4 cores online) and
  `sudo jetson_clocks` (pins CPU frequency, so frequency scaling does not
  add noise). In the 5 W mode (`nvpmodel -m 1`) only 2 cores are online and
  the always-4 policy is meaningless.
- **Power supply.** Use the barrel-jack supply. On micro-USB the board can
  throttle under 4-core load.
- **Nothing else running.** No builds, no desktop session if avoidable. The
  logger itself wakes 100 times a second; that load is the same for every
  policy, but it is not zero.
- **OpenMP wait policy.** After a parallel region, libgomp worker threads
  spin briefly before sleeping, which draws power. `run_measure.sh` leaves
  `OMP_WAIT_POLICY` as you set it; try `OMP_WAIT_POLICY=passive
  ./scripts/measure/run_measure.sh ...` as a second configuration and
  report which one you used.
- **Repeat the run.** One run gives one number per image with no error
  bar. Run it at least three times and report the spread.

## Power sources

`power_logger.py` auto-detects, in this order:

1. `ina3221x` — original Jetson Nano on JetPack 4: reads
   `/sys/bus/i2c/drivers/ina3221x/*/iio:device*/in_power*_input`. Main rail
   `POM_5V_IN` (whole-board input power); `POM_5V_CPU` and `POM_5V_GPU` are
   logged too.
2. `hwmon` — JetPack 5+ boards (e.g. Orin Nano): INA3221 via hwmon, main
   rail `VDD_IN`.
3. `tegrastats` — fallback, parses `tegrastats` output. Coarser timing.
4. `--file PATH --file-scale K` — any file holding one number, for other
   boards.

Check what it found before a long run:

```bash
python3 scripts/measure/power_logger.py --probe
```

To analyse a different rail from an existing log (e.g. CPU only):

```bash
python3 scripts/measure/analyze_energy.py bench_results/measure/phases.csv \
  bench_results/measure/power.csv --rail cpu
```

The Python scripts use only the standard library and avoid syntax newer
than Python 3.6, so the Jetson's system `python3` is enough.

## What has and has not been tested

Developed on a Mac, without a Jetson to hand:

- **Tested:** `jcadaptivebench-static` builds and runs; the overhead
  analysis end to end on DIV2K images; the energy analysis end to end
  against a fake constant-power file (`--file`); the integration maths and
  the `tegrastats` line parser on sample inputs.
- **Not tested on real hardware:** the `ina3221x` and `hwmon` sysfs paths
  and the live `tegrastats` fallback. Run `power_logger.py --probe` on the
  board first; if it reports no source, find the sensor's power file and
  pass it with `--file`.
- **Not run under Python 3.6**, only written to be compatible with it.
