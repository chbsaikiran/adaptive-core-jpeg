# Building and running the adaptive-core JPEG encoder on Jetson Nano

End-to-end steps to build this fork and run `jcadaptive-static` (the
classifier-driven 2-vs-4-core JPEG encoder — see
[Steps_to_build_and_benchmark_mac.md](Steps_to_build_and_benchmark_mac.md)
section 6 for what it does) on an NVIDIA Jetson Nano.

This assumes the Jetson Nano already has a working JetPack/L4T (Ubuntu-based)
image flashed and you can reach it over SSH or a local terminal — flashing
the board itself is out of scope here.

**Important, read first:** the original Jetson Nano's CPU is a **quad-core**
ARM Cortex-A57 (4 cores total, no more). "4 cores" for a complex image is
therefore *every* CPU core on the board, not a subset — there's no headroom
left for anything else (camera capture, other processes) while a "complex"
image encodes. "2 cores" leaves the other 2 free. Keep this in mind when
deciding whether the adaptive-core tradeoff is actually useful for your
workload on this specific board. (Newer "Jetson Orin Nano" boards have a
6-core CPU — same idea, different core count.)

---

## 1. Software needed

Nothing exotic — JetPack's default toolchain covers everything. From a
terminal on the Jetson:

```bash
sudo apt update
sudo apt install -y build-essential cmake git
```

- `build-essential` gives you `gcc`/`g++`, `make`, and **OpenMP support is
  already built into GCC** (`libgomp`) — unlike on macOS/Clang, there's no
  separate `libomp` package to install here.
- NASM/Yasm is **not** needed — this is ARM64, so SIMD is built from
  `simd/arm/` directly by the C compiler (same as Apple Silicon in the Mac
  build doc).
- No Python, no `uv`, no scikit-learn needed on the Jetson at all — the
  trained classifier is already compiled into `src/complexity_model.c` and
  ships as plain C with zero runtime dependencies (see
  `scripts/ml/README.md` section 6 for how it got there). Python is only
  needed on your dev machine if you retrain the model later (step 6).

## 2. Get the code onto the Jetson

Whichever is easiest for your setup:

```bash
# Option A: clone directly on the Jetson (if it has a route to your remote)
git clone <your-repo-url> adaptive-core-jpeg
cd adaptive-core-jpeg

# Option B: copy from your dev machine over SSH
rsync -avz --exclude build --exclude '*.venv' \
  /path/to/adaptive-core-jpeg/ jetson@<jetson-ip>:~/adaptive-core-jpeg/
```

`bench_results/`, `testimages/DIV2K*`, and `.venv/` are all gitignored and
not needed on-device — skip them if copying manually (the `rsync
--exclude`s above, or a plain `git clone`, both do this for you).

## 3. Configure and build

```bash
cd adaptive-core-jpeg
mkdir -p build && cd build
cmake -DWITH_OPENMP=1 ..
```

Check the configure output for these two lines, confirming OpenMP and SIMD
were actually found (not silently disabled):
```
-- OpenMP-parallel strip encoder (jcparallel) enabled (WITH_OPENMP = 1)
-- SIMD extensions: ARM64 (WITH_SIMD = 1)
```

Build the targets you need:
```bash
cmake --build . --target jcparalleltest-static jcadaptive-static -- -j"$(nproc)"
```

`-j"$(nproc)"` uses all 4 cores to build — the Nano's CPU is slow compared
to a dev laptop, so this can take a few minutes. If the board is the 2GB
RAM variant and the build gets killed (OOM) or grinds to a crawl, add swap
first:
```bash
sudo fallocate -l 4G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile
```

## 4. Correctness check

Run this once after building, before trusting any output:

```bash
./jcparalleltest-static
# or: ctest -R jcparalleltest --output-on-failure
```
Expect `ALL PASS`. This decodes both the per-strip and the spliced output
and byte-compares against a single-threaded reference across thread counts
1-4 — see `src/jcparallel.c`/`.h` for what it's actually checking.

## 5. Run the adaptive-core encoder

```bash
./jcadaptive-static <input-image> <output.jpg> [-quality Q] [-threads N]
```
- Supported input formats: JPEG (8-bit), BMP, PNG, PPM/PGM.
- `-quality Q` — JPEG quality, 0-100 (default 85).
- `-threads N` — override the classifier and force that core count (for
  comparison; normal use omits this).

Example:
```bash
$ ./jcadaptive-static ../testimages/vgl_5674_0098.png out.jpg
../testimages/vgl_5674_0098.png: 120x96, predicted complex (P(simple)=0.000 P(complex)=1.000) -> 4 cores
out.jpg: wrote 9223 bytes (4 strips spliced, 4 cores)
```

### Viewing/verifying the output

The Jetson is usually headless (SSH-only), so the easiest way to *look at*
the result is to copy it back to a machine with a display:
```bash
scp jetson@<jetson-ip>:~/adaptive-core-jpeg/build/out.jpg .
open out.jpg   # macOS; use your OS's image viewer otherwise
```

If you also built `cjpeg-static`/`djpeg-static`
(`cmake --build . --target cjpeg-static djpeg-static`), you can get the
same objective proof used during development — that the adaptive/spliced
encode decodes pixel-identical to a normal single-threaded encode — right
there on the Jetson, no display needed:
```bash
./cjpeg-static -quality 85 -outfile ref.jpg ../testimages/vgl_5674_0098.png
./djpeg-static -outfile ref.ppm ref.jpg
./djpeg-static -outfile adaptive.ppm out.jpg
cmp ref.ppm adaptive.ppm && echo "IDENTICAL PIXELS" || echo "DIFFER"
```

## 6. (Optional) Re-benchmarking and retraining for this specific board

The shipped classifier (`src/complexity_model.c`) was trained on 2-vs-4-core
timing data collected on a different machine (see `scripts/ml/README.md`).
CPU microarchitecture affects the *relative* speedup of 4 cores over 2, so
its "simple"/"complex" predictions carry over reasonably (the underlying
image features it looks at — entropy, edge density, DCT energy — aren't
hardware-dependent) but aren't guaranteed optimal for the Nano's specific
Cortex-A57 cores. To get a Nano-accurate model:

1. On the Jetson, build `jcparallelbench-static` and run
   `./scripts/bench_2vs4.sh <dataset_dir> <repeat>` (see
   `Steps_to_build_and_benchmark_mac.md` section 4 — same tool, same steps,
   just running on the Nano this time) to collect fresh 2-vs-4-core timing
   data on this hardware.
2. Copy the resulting `bench_results/dataset_train.csv` /
   `dataset_val.csv` back to your dev machine.
3. On your dev machine (needs Python + `uv`, not the Jetson), re-run the
   training pipeline in `scripts/ml/` (`build_dataset.py` →
   `extract_features.py` → `train_model.py` → `export_c_model.py --out
   ../../src/complexity_model.c` — see `scripts/ml/README.md`).
4. Copy the regenerated `src/complexity_model.c` back to the Jetson and
   rebuild `jcadaptive-static` (step 3 above).

## Notes / caveats

- **Feature extraction fidelity:** `src/jcfeatures.c` (the C port of
  `scripts/ml/extract_features.py`) is a close but not bit-exact port —
  see its header comment. This doesn't change with platform; it applies
  equally on the Jetson.
- **Memory:** the 2GB Jetson Nano variant can struggle to *build* this
  (see the swap note in step 3); once built, `jcadaptive-static` itself
  processes one image at a time and doesn't need much memory.
- **`-j"$(nproc)"` while also benchmarking:** if you're running
  `jcparallelbench-static`/`jcadaptive-static` timing comparisons on the
  Nano, don't do it while also compiling something else in the background
  — with only 4 cores total, that will visibly skew the 2-vs-4-core
  timing comparison.
