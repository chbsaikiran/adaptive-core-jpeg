# Building and benchmarking `jcparallel` on Mac (2-core vs 4-core)

End-to-end steps for this fork's experimental strip-parallel JPEG encoder
(`src/jcparallel.c`/`.h`) on macOS (Apple Silicon, arm64): install the build
tools, build it, then produce a per-image dataset comparing 2-core vs 4-core
encode time (used downstream to label images "simple"/"complex" for training
a core-count-selection classifier).

Companion doc: [Steps_to_build_multi_core.txt](Steps_to_build_multi_core.txt)
covers the original Windows/MSVC build; this doc is the Mac equivalent, plus
the benchmarking pipeline that doesn't exist on the Windows side yet.

---

## 1. Software needed

| Tool | Why | Install |
|---|---|---|
| Xcode Command Line Tools | `clang`/`gcc` | usually already present (`clang --version` to check); `xcode-select --install` if not |
| [Homebrew](https://brew.sh) | installs the two packages below | — |
| CMake ≥ 3.15 | project's build system | `brew install cmake` |
| libomp | OpenMP runtime — Apple Clang has no built-in OpenMP support | `brew install libomp` |

NASM/Yasm is **not** needed here — it's only required for x86/x86-64 SIMD;
on Apple Silicon, SIMD is built from `simd/arm` directly by the C compiler.

```bash
brew install cmake libomp
```

## 2. Configure and build

From the repo root:

```bash
rm -rf build
mkdir build && cd build
cmake -DWITH_OPENMP=1 -DOpenMP_ROOT=$(brew --prefix libomp) ..
```

Check the configure output for these two lines — they confirm OpenMP and
SIMD were actually found (not silently disabled):
```
-- OpenMP-parallel strip encoder (jcparallel) enabled (WITH_OPENMP = 1)
-- SIMD extensions: ARM64 (WITH_SIMD = 1)
```

If OpenMP isn't found even with `-DOpenMP_ROOT=...`, fall back to explicit
flags:
```bash
cmake -DWITH_OPENMP=1 \
  -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I$(brew --prefix libomp)/include" \
  -DOpenMP_C_LIB_NAMES="omp" \
  -DOpenMP_omp_LIBRARY="$(brew --prefix libomp)/lib/libomp.dylib" ..
```

Build the two tools used below (run from `build/`, or let the scripts in
step 4 do this for you automatically):

```bash
cmake --build . --target jcparalleltest-static  -- -j"$(sysctl -n hw.ncpu)"
cmake --build . --target jcparallelbench-static -- -j"$(sysctl -n hw.ncpu)"
```

## 3. (Optional) Correctness check

`jcparalleltest-static` is a smoke test — one synthetic image, checked for
pixel-exact output across thread counts 1/2/3/4. Worth running once after
any change to `jcparallel.c`:

```bash
./jcparalleltest-static
# or: ctest -R jcparalleltest --output-on-failure
```
Expect `ALL PASS`, exit code 0.

## 4. Get per-image 2-core vs 4-core timing (the actual benchmark)

This is what `jcparallelbench-static` + the `scripts/` wrappers are for —
see [src/jcparallelbench.c](src/jcparallelbench.c) for full design notes.
It decodes each image once, then times only the
`jpar_encode_strips_parallel()` call (wall-clock, `CLOCK_MONOTONIC`,
repeated many times per image for a stable reading), and never times file
I/O or decode.

From the repo root (not `build/`):

```bash
./scripts/bench_2vs4.sh [dataset_dir] [repeat]
```
- `dataset_dir` — folder of images to benchmark. Defaults to the repo's
  `testimages/` (a handful of small sample images — fine to validate the
  pipeline, too few/small to mean anything for real classifier training).
- `repeat` — encodes per image per thread count. Defaults to `50`. Bump
  this up for very small/fast images so the timing isn't dominated by
  scheduler noise.

Example, pointing at a real photo folder:
```bash
./scripts/bench_2vs4.sh ~/Pictures/my-dataset 50
```

This runs three steps for you:
1. `scripts/run_bench.sh 2 <dataset_dir> <repeat>` → builds
   `jcparallelbench-static` if needed, runs it at `--threads 2`, saves
   `bench_results/threads_2.csv`.
2. `scripts/run_bench.sh 4 <dataset_dir> <repeat>` → same at `--threads 4`,
   saves `bench_results/threads_4.csv`.
3. `scripts/merge_bench.py` joins the two CSVs on image filename into
   `bench_results/dataset.csv` — **this is the deliverable.**

### Output: `bench_results/dataset.csv`

```
image,width,height,mean_2core_s,min_2core_s,mean_4core_s,min_4core_s,speedup_min
testorig.jpg,227,149,0.000033,0.000025,0.000028,0.000024,1.0417
...
```
- `mean_*_s` / `min_*_s` — average and minimum encode time (seconds) over
  `repeat` runs at that thread count. `min` is the standard low-noise
  benchmarking metric; `mean` is closer to real deployed latency.
- `speedup_min = min_2core_s / min_4core_s` — the column to threshold for
  labeling: meaningfully > 1 → 4 cores helped → label "complex"; close to
  or below 1 → label "simple".

Each individual run's raw per-image CSV (before merging) is also kept:
`bench_results/threads_2.csv`, `bench_results/threads_4.csv`.

## 5. Re-running after code changes

The scripts always rebuild `jcparallelbench-static` before running, so
after editing `jcparallel.c`/`jcparallelbench.c` you can just re-run step 4
directly — no need to reconfigure CMake unless `CMakeLists.txt` itself
changed.

## Notes / caveats

- **Supported input formats:** JPEG (8-bit only — 12-/16-bit precision
  files are skipped with a warning), BMP, PPM/PGM. PNG/GIF/Targa and
  non-image files are silently skipped, so pointing `dataset_dir` at a
  folder with stray non-image files is safe.
- **Small images are noisy.** On `testimages/` (≤230×230px), a single
  encode is sub-millisecond and `speedup_min` swings from ~0.96 to ~1.41
  across images purely from scheduling/thread-launch overhead — that's
  expected at this scale, not a bug. A real photo dataset (larger images,
  hundreds+ of them) will give a much more meaningful signal.
- **Apple Silicon P/E cores:** OpenMP's `num_threads` picks how many
  strips run concurrently, but doesn't pin threads to specific physical
  cores — on a heterogeneous P-core/E-core Mac this can add its own noise
  to the 2-vs-4 comparison, independent of image complexity.
