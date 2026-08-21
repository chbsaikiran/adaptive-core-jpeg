#!/usr/bin/env bash
#
# bench_2vs4.sh [dataset_dir] [repeat]
#
# Runs jcparallelbench-static over dataset_dir once at --threads 2 and once
# at --threads 4, then merges the two per-image CSVs into
# bench_results/dataset.csv -- the actual per-image (2-core time, 4-core
# time) dataset for the complexity-labeling/classifier-training step.
#
# dataset_dir defaults to the repo's testimages/ (a handful of small images,
# good for validating this pipeline end-to-end, not for training on -- see
# the plan/README notes). repeat defaults to 50 encodes/image, since these
# images are small enough that a single encode is sub-millisecond/noisy.

set -euo pipefail

DATASET_DIR="${1:-testimages}"
REPEAT="${2:-50}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$REPO_ROOT/bench_results"

"$SCRIPT_DIR/run_bench.sh" 2 "$DATASET_DIR" "$REPEAT"
"$SCRIPT_DIR/run_bench.sh" 4 "$DATASET_DIR" "$REPEAT"

python3 "$SCRIPT_DIR/merge_bench.py" \
  "$RESULTS_DIR/threads_2.csv" "$RESULTS_DIR/threads_4.csv" \
  -o "$RESULTS_DIR/dataset.csv"

echo
echo "Per-image 2-core-vs-4-core dataset: $RESULTS_DIR/dataset.csv"
