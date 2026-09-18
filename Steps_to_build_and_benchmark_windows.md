# Building and benchmarking `jcparallel` on Windows (2-core vs 4-core)

End-to-end steps for this fork's experimental strip-parallel JPEG encoder
(`src/jcparallel.c`/`.h`) on Windows (x64, MSVC): install the build tools,
build it, run the correctness test, then produce a per-image dataset
comparing 2-core vs 4-core encode time (used downstream to label images
"simple"/"complex" for training a core-count-selection classifier).

Companion docs:
[Steps_to_build_multi_core.txt](Steps_to_build_multi_core.txt) is the
original bare build-command list; [Steps_to_build_and_benchmark_mac.md](Steps_to_build_and_benchmark_mac.md)
is the macOS equivalent of *this* doc. The benchmarking pipeline is the
same on both platforms -- only the wrapper scripts differ (`scripts\*.ps1`
here, `scripts/*.sh` on Mac).

---

## 1. Software needed

| Tool | Why | Install |
|---|---|---|
| Visual Studio 2022 or 2026, **"Desktop development with C++"** workload | MSVC compiler + its built-in OpenMP (`/openmp`) | <https://visualstudio.microsoft.com/> |
| CMake ≥ 3.15 | project's build system | Bundled inside VS (`...\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe`), or standalone from <https://cmake.org/download/> (tick "Add to PATH") |
| Python 3 | `scripts\merge_bench.py` (final merge step only) | <https://www.python.org/downloads/> (tick "Add python.exe to PATH"), or `winget install Python.Python.3.12` |

**NASM/Yasm is not needed** -- these steps pass `-DWITH_SIMD=0`, so there
is no assembly to build. (SIMD only accelerates the *baseline* single-image
encode path; it is orthogonal to the strip-parallel work being measured
here.)

MSVC's OpenMP is version 2.0, which is old but implements everything
`jcparallel.c` uses (`#pragma omp parallel for num_threads(...)`).

### Check your toolchain

```powershell
cmake --version                       # if "not recognized", use the full path to the VS-bundled cmake.exe
python --version                      # 3.x
& "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe" -latest -property displayName
```

If `cmake` isn't on `PATH`, either add it or set a variable you'll reuse
below (adjust the VS year/edition):

```powershell
$cmake = "C:\Program Files\Microsoft Visual Studio\18\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe"
# then use  & $cmake ...  everywhere this doc writes  cmake ...
```

## 2. One-time: allow PowerShell to run the wrapper scripts

Fresh Windows installs block local `.ps1` scripts. Enable them for your
user account (no admin needed; still blocks unsigned scripts downloaded
from the internet):

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

To undo later: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy Undefined`.
Alternatively, skip this and invoke each script as
`powershell -ExecutionPolicy Bypass -File .\scripts\bench_2vs4.ps1`.

## 3. Configure and build

From the repo root. Pick the generator matching your VS version --
`cmake --help` lists the exact strings ("Visual Studio 17 2022",
"Visual Studio 18 2026", ...).

```powershell
Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
cmake -G "Visual Studio 18 2026" -A x64 -DWITH_SIMD=0 -DWITH_OPENMP=1 -DWITH_JPEG7=0 -S . -B build
```

Confirm OpenMP was actually found in the configure output (not silently
disabled):

```
-- OpenMP-parallel strip encoder (jcparallel) enabled (WITH_OPENMP = 1)
```

If it instead prints `WITH_OPENMP was requested, but no OpenMP C support
was found`, your VS C++ workload is incomplete -- reinstall it via the
Visual Studio Installer. Without OpenMP the encoder still builds but runs
the strips **sequentially**, so a 2-vs-4 comparison would show no
difference.

Build the two tools (Release config):

```powershell
cmake --build build --config Release --target jcparalleltest-static  --parallel
cmake --build build --config Release --target jcparallelbench-static --parallel
```

Executables land in `build\Release\`:
`jcparalleltest-static.exe`, `jcparallelbench-static.exe`.

> The `scripts\` wrappers in step 5 rebuild `jcparallelbench-static`
> themselves, so after the first configure you normally don't run these
> build commands by hand.

## 4. Correctness check

`jcparalleltest-static` is a smoke test -- one synthetic image, checked for
pixel-exact output against a single-threaded reference across thread counts
1/2/3/4. Worth running once after any change to `jcparallel.c`.

```powershell
.\build\Release\jcparalleltest-static.exe
# or, via CTest:
ctest --test-dir build -C Release -R jcparalleltest --output-on-failure
```

Expect `ALL PASS` and exit code 0.

## 5. Get per-image 2-core vs 4-core timing (the actual benchmark)

This is what `jcparallelbench-static` + the `scripts\` wrappers are for --
see [src/jcparallelbench.c](src/jcparallelbench.c) for full design notes.
It decodes each image once, then times **only** the
`jpar_encode_strips_parallel()` call (wall-clock, repeated many times per
image for a stable reading) -- never file I/O or decode.

"2-core" and "4-core" here mean `--threads 2` and `--threads 4`: the image
is cut into that many horizontal strips and the strips are encoded
concurrently on that many OpenMP threads. OpenMP does not pin threads to
physical cores, so on a busy machine the counts are "up to N-way parallel",
not a hard core reservation.

From the repo root:

```powershell
.\scripts\bench_2vs4.ps1 [dataset_dir] [repeat] [-Config Release]
```

- `dataset_dir` -- folder of images to benchmark. Defaults to the repo's
  `testimages\` (a handful of tiny sample images -- fine to validate the
  pipeline, far too few/small to mean anything for real classifier
  training).
- `repeat` -- encodes per image per thread count. Defaults to `50`. Bump
  it up for very small/fast images so timing isn't dominated by
  thread-launch/scheduler noise.
- `-Config` -- build config to build/run. Defaults to `Release`. Use
  `Release` for any real numbers; `Debug` is far slower and noisier.

Example against a real photo folder:

```powershell
.\scripts\bench_2vs4.ps1 C:\datasets\photos 50
```

It runs three steps:

1. `scripts\run_bench.ps1 2 <dataset_dir> <repeat>` -> builds
   `jcparallelbench-static` if needed, runs it at `--threads 2`, writes
   `bench_results\threads_2.csv`.
2. `scripts\run_bench.ps1 4 <dataset_dir> <repeat>` -> same at
   `--threads 4`, writes `bench_results\threads_4.csv`.
3. `scripts\merge_bench.py` joins the two CSVs on image filename into
   `bench_results\dataset.csv` -- **this is the deliverable.**

You can also run a single thread count directly:

```powershell
.\scripts\run_bench.ps1 4 testimages 50
```

### Output: `bench_results\dataset.csv`

```
image,width,height,mean_2core_s,min_2core_s,mean_4core_s,min_4core_s,speedup_min
monkey16.pgm,149,227,0.000266,0.000241,0.000141,0.000113,2.1327
testorig.jpg,227,149,0.000373,0.000317,0.000345,0.000223,1.4215
testimgint.jpg,227,149,0.000340,0.000316,0.000429,0.000386,0.8187
...
```

- `mean_*_s` / `min_*_s` -- average and minimum encode time (seconds) over
  `repeat` runs at that thread count. `min` is the standard low-noise
  benchmarking metric; `mean` is closer to real deployed latency.
- `speedup_min = min_2core_s / min_4core_s` -- the column to threshold for
  labeling: meaningfully > 1 -> 4 cores helped -> label **"complex"**;
  close to or below 1 -> label **"simple"**.

The two raw per-thread-count CSVs (`bench_results\threads_2.csv`,
`bench_results\threads_4.csv`) are kept as well.

`bench_results\` is git-ignored -- it's machine- and dataset-specific,
regenerate it on demand.

## 6. Re-running after code changes

The `scripts\` wrappers always rebuild `jcparallelbench-static` before
running, so after editing `jcparallel.c` / `jcparallelbench.c` you can just
re-run step 5. You only need to re-run the step 3 `cmake` *configure* if
`CMakeLists.txt` itself changed.

## Notes / caveats

- **Supported input formats:** JPEG (8-bit only -- 12-/16-bit precision
  files are skipped with a `skip ...` note on stderr), BMP, PPM/PGM.
  PNG/GIF/Targa and non-image files are silently skipped, so pointing
  `dataset_dir` at a folder that also has stray files is safe. (In
  `testimages\`, `monkey12.jpg` is a 12-bit file and is expected to be
  skipped.)
- **Small images are noisy.** On `testimages\` (≤230×230 px) a single
  encode is sub-millisecond and `speedup_min` swings wildly (≈0.8 to ≈2.1)
  purely from thread-launch/scheduler overhead -- that's expected at this
  scale, not a bug. A real dataset (larger images, hundreds+ of them)
  gives a meaningful signal.
- **`Debug` builds** are much slower and much noisier -- always benchmark
  with `-Config Release` (the default).
- **stderr handling:** the exe prints per-image `skip ...` notes and
  `merge_bench.py` prints its summary to stderr. Windows PowerShell would
  normally treat a native command's stderr as a fatal error; the wrapper
  scripts already handle this (they route stderr to the console and gate
  only on the real exit code), so those lines are informational, not
  failures.
- **CSV encoding:** the scripts write the CSVs as UTF-8 without a BOM (via
  .NET), because PowerShell's own `Tee-Object`/`Set-Content` would emit
  UTF-16 or a BOM that `merge_bench.py`'s CSV reader chokes on.
- **CPU count:** `run_bench.ps1` builds with `--parallel` using
  `%NUMBER_OF_PROCESSORS%`. That only affects *build* speed; the *encode*
  parallelism is set purely by the `--threads` value.
