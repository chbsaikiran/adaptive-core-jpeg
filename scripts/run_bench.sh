#!/usr/bin/env bash
#
# run_bench.sh <threads> [dataset_dir] [repeat]
#
# Builds jcparallelbench-static (if needed) and runs it over dataset_dir at
# a fixed thread count, saving its per-image CSV output to
# bench_results/threads_<N>.csv. See jcparallelbench.c for what's measured
# (encode-only wall-clock time, mean+min over `repeat` encodes per image).
#
# Typically invoked via bench_2vs4.sh rather than directly.

set -euo pipefail

THREADS="${1:?Usage: run_bench.sh <threads> [dataset_dir] [repeat]}"
DATASET_DIR="${2:-testimages}"
REPEAT="${3:-50}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
RESULTS_DIR="$REPO_ROOT/bench_results"

# Resolve a relative dataset_dir against the repo root (not the caller's
# cwd), so `./scripts/run_bench.sh 4` works the same from anywhere.
if [[ "$DATASET_DIR" != /* ]]; then
  DATASET_DIR="$REPO_ROOT/$DATASET_DIR"
fi

if [[ ! -d "$BUILD_DIR" ]]; then
  echo "error: $BUILD_DIR not found -- configure it first (see" \
       "Steps_to_build_multi_core.txt / the prior Mac build session:" \
       "cmake -DWITH_OPENMP=1 -DOpenMP_ROOT=\$(brew --prefix libomp) -B build)" >&2
  exit 1
fi

mkdir -p "$RESULTS_DIR"

NPROC="$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)"
cmake --build "$BUILD_DIR" --target jcparallelbench-static -- -j"$NPROC"

OUT_CSV="$RESULTS_DIR/threads_${THREADS}.csv"
"$BUILD_DIR/jcparallelbench-static" "$DATASET_DIR" --threads "$THREADS" \
  --repeat "$REPEAT" | tee "$OUT_CSV"

echo "Wrote $OUT_CSV" >&2
