#!/usr/bin/env bash
#
# run_measure.sh [dataset_dir] [min_seconds] [limit]
#
# One-shot driver for the two measurements in this directory:
#   - energy per image for always-2-core, always-4-core and adaptive
#   - classifier overhead as a share of adaptive encode time
#
# Builds jcadaptivebench-static (if needed), starts power_logger.py in the
# background, runs the benchmark over dataset_dir, stops the logger, then
# runs analyze_overhead.py and analyze_energy.py. Everything lands in
# bench_results/measure/.
#
# dataset_dir defaults to testimages/DIV2K_valid_HR (the 100-image split the
# classifier was *not* trained on). min_seconds is how long each phase is
# stretched (default 2 -- see jcadaptivebench.c for why). limit caps the
# number of images, for a quick trial run.
#
# With no power sensor (i.e. anywhere but a Jetson), the energy step is
# skipped and only the classifier-overhead numbers are produced.
#
# Extra power_logger.py arguments (e.g. "--file /path --file-scale 0.001",
# or "--source tegrastats") can be passed via POWER_LOGGER_ARGS.

set -euo pipefail

DATASET_DIR="${1:-testimages/DIV2K_valid_HR}"
MIN_SECONDS="${2:-2}"
LIMIT="${3:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
RESULTS_DIR="$REPO_ROOT/bench_results/measure"

if [[ "$DATASET_DIR" != /* ]]; then
  DATASET_DIR="$REPO_ROOT/$DATASET_DIR"
fi

if [[ ! -d "$BUILD_DIR" ]]; then
  echo "error: $BUILD_DIR not found -- configure it first (see" \
       "Steps_to_build_and_run_jetson_nano.md section 3:" \
       "cmake -DWITH_OPENMP=1 -B build)" >&2
  exit 1
fi
if [[ ! -d "$DATASET_DIR" ]]; then
  echo "error: dataset directory $DATASET_DIR not found" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

NPROC="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
cmake --build "$BUILD_DIR" --target jcadaptivebench-static -- -j"$NPROC"

PHASES_CSV="$RESULTS_DIR/phases.csv"
POWER_CSV="$RESULTS_DIR/power.csv"
META="$RESULTS_DIR/run_info.txt"

# shellcheck disable=SC2206  # intentional word-splitting of extra args
LOGGER_ARGS=(${POWER_LOGGER_ARGS:-})

# Record the conditions the numbers were taken under -- power mode and
# OpenMP wait policy both move the energy results, so they belong next to
# the data.
{
  echo "date: $(date)"
  echo "host: $(uname -a)"
  echo "dataset: $DATASET_DIR"
  echo "min_seconds: $MIN_SECONDS"
  echo "limit: ${LIMIT:-all}"
  echo "OMP_WAIT_POLICY: ${OMP_WAIT_POLICY:-<unset>}"
  echo "nvpmodel: $(nvpmodel -q 2>/dev/null | tr '\n' ' ' || true)"
} > "$META"

LOGGER_PID=""
stop_logger() {
  if [[ -n "$LOGGER_PID" ]]; then
    kill -TERM "$LOGGER_PID" 2>/dev/null || true
    wait "$LOGGER_PID" 2>/dev/null || true
    LOGGER_PID=""
  fi
}
trap stop_logger EXIT

HAVE_POWER=0
if python3 "$SCRIPT_DIR/power_logger.py" --probe "${LOGGER_ARGS[@]+"${LOGGER_ARGS[@]}"}" \
     >> "$META" 2>/dev/null; then
  HAVE_POWER=1
  rm -f "$POWER_CSV"
  python3 "$SCRIPT_DIR/power_logger.py" -o "$POWER_CSV" \
    "${LOGGER_ARGS[@]+"${LOGGER_ARGS[@]}"}" &
  LOGGER_PID=$!
  sleep 1   # let the logger take its first samples before phase 1 starts
else
  echo "note: no power source found -- measuring classifier overhead only" \
       "(energy needs a Jetson's power sensor; see README.md)" >&2
fi

BENCH_ARGS=("$DATASET_DIR" --min-seconds "$MIN_SECONDS")
if [[ -n "$LIMIT" ]]; then
  BENCH_ARGS+=(--limit "$LIMIT")
fi
"$BUILD_DIR/jcadaptivebench-static" "${BENCH_ARGS[@]}" > "$PHASES_CSV"

stop_logger

echo
echo "===== Classifier overhead ====="
python3 "$SCRIPT_DIR/analyze_overhead.py" "$PHASES_CSV" \
  -o "$RESULTS_DIR/overhead_per_image.csv" | tee "$RESULTS_DIR/overhead_summary.txt"

if [[ "$HAVE_POWER" == 1 ]]; then
  echo
  echo "===== Energy per image ====="
  python3 "$SCRIPT_DIR/analyze_energy.py" "$PHASES_CSV" "$POWER_CSV" \
    -o "$RESULTS_DIR/energy_per_image.csv" | tee "$RESULTS_DIR/energy_summary.txt"
fi

echo
echo "Results in $RESULTS_DIR" >&2
