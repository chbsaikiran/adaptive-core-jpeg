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

## 6. Adaptive-core encoding (`jcadaptive-static`)

This is the actual production path: given one real image, classify it as
"simple" or "complex" (the model trained from step 4's `dataset.csv` — see
[scripts/ml/README.md](scripts/ml/README.md)) and encode it with 2 or 4
cores accordingly, producing one real, spliced `.jpg` file (not per-strip
files — `jpar_splice_strips()` in `src/jcparallel.c` merges the strips via
restart markers into a single valid JPEG).

```bash
cmake --build build --target jcadaptive-static -- -j"$(sysctl -n hw.ncpu)"
./build/jcadaptive-static <input-image> <output.jpg> [-quality Q] [-threads N]
```
- `-quality Q` — JPEG quality, 0-100 (default 85).
- `-threads N` — override the classifier and force that core count (for
  comparison; normal use omits this and lets the classifier decide).

Example:
```bash
$ ./build/jcadaptive-static testimages/vgl_5674_0098.png /tmp/out.jpg
testimages/vgl_5674_0098.png: 120x96, predicted complex (P(simple)=0.000 P(complex)=1.000) -> 4 cores
/tmp/out.jpg: wrote 9223 bytes (4 strips spliced, 4 cores)
```

### Decoding the output (visual proof it's a real, correct JPEG)

`jcadaptive-static`'s output is one ordinary, standalone JPEG file — any
JPEG decoder can open it, including macOS Preview/QuickLook. Two ways to
look at it:

**Quickest — just open it:**
```bash
open /tmp/out.jpg          # opens in Preview
qlmanage -p /tmp/out.jpg   # or QuickLook, from the terminal
```

**Decode with this repo's own decoder** (`djpeg-static`, built alongside
`cjpeg-static` — build both if you haven't: `cmake --build build --target
cjpeg-static djpeg-static`) and compare side-by-side against the original:
```bash
./build/djpeg-static -png -outfile /tmp/out.png /tmp/out.jpg
open testimages/vgl_5674_0098.png /tmp/out.png   # opens both in Preview
```
At quality 85 the two should look visually indistinguishable (JPEG is
lossy, so they won't be byte-identical pixel data, but there should be no
visible artifacting, banding, or strip seams — a strip seam specifically
would indicate the restart-marker splice is broken, since it would show up
as a visible discontinuity at the strip boundary rows).

**Objective proof, not just "looks the same":** compare against a normal
single-threaded encode of the same source image — if `jcadaptive-static`'s
adaptive core selection and splicing are correct, the decoded pixels
should match *exactly* (JPEG decode is deterministic, so two encodes at
the same quality that produce the same compressed data decode to
identical pixels; jcparallel's restart-marker splice is specifically
designed to reproduce a standard single-threaded encode's output — see
`src/jcparallel.h`):
```bash
./build/cjpeg-static -quality 85 -outfile /tmp/ref.jpg testimages/vgl_5674_0098.png
./build/djpeg-static -outfile /tmp/ref.ppm /tmp/ref.jpg
./build/djpeg-static -outfile /tmp/adaptive.ppm /tmp/out.jpg
cmp /tmp/ref.ppm /tmp/adaptive.ppm && echo "IDENTICAL PIXELS" || echo "DIFFER"
```
This is exactly the check used to validate the splice implementation
during development (see the "Verified" note below) — `cmp` printing
nothing and `IDENTICAL PIXELS` is the proof the two encodes decode to the
same image.

Under the hood: `src/jcfeatures.c` computes the same 9 pre-encode pixel
features as `scripts/ml/extract_features.py` (entropy, edge density/DCT
energy, per-channel variance, etc. — see its header comment for the
feature list and a note on numerical fidelity vs. the Python version),
`src/complexity_model.c` (the trained Random Forest, exported from Python
via `scripts/ml/export_c_model.py`) scores them, and the predicted label
picks 2 vs 4 threads for `jpar_encode_strips_spliced()`.

Verified: `jcparalleltest-static` decodes both the per-strip *and* spliced
output and byte-compares against a single-threaded reference across thread
counts 1-4 (`ctest -R jcparalleltest`); the spliced path was additionally
checked against `cjpeg`/`djpeg` on real (non-synthetic) images of various
sizes, pixel-identical in every case tried.

## Notes / caveats

- **Supported input formats:** JPEG (8-bit only — 12-/16-bit precision
  files are skipped with a warning), BMP, PNG, PPM/PGM. GIF/Targa and other
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
