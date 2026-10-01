# Guide to the executables in this project

This repository is a fork of libjpeg-turbo with an adaptive-core JPEG
encoder added on top. Building it produces several executables in the
`build/` folder. This guide says what each one is for and which one to run.

**If you only want to run the project: use `jcadaptive-static`.** It is the
only executable that runs the full pipeline (model first, then the encoder
on 2 or 4 cores according to the model's answer).

## At a glance

| Executable | Purpose | Runs the model? | Writes a JPEG? |
|---|---|---|---|
| `jcadaptive-static` | **The full pipeline**, one image in, one JPEG out | Yes | Yes |
| `jcadaptivebench-static` | Measures time (and, with the scripts, energy) of each policy over a folder of images | Yes | No |
| `jcparallelbench-static` | Times the parallel encoder at one fixed core count; produces the training data for the model | No | No |
| `jcparalleltest-static` | Checks the parallel encoder produces correct output | No | No |
| `cjpeg-static`, `djpeg-static` | Stock libjpeg-turbo encoder and decoder, used as the reference | No | Yes (`cjpeg`) |

All of them are built from the `build/` folder. Configure once, then build
whichever targets you need:

```bash
cmake -DWITH_OPENMP=1 -B build
cmake --build build --target jcadaptive-static jcadaptivebench-static \
  jcparallelbench-static jcparalleltest-static cjpeg-static djpeg-static
```

Platform-specific setup (OpenMP on macOS, Windows, Jetson Nano) is in the
`Steps_to_build_*.md` files in this folder.

---

## `jcadaptive-static` — the full pipeline

Source: [src/jcadaptive.c](src/jcadaptive.c)

For one input image it does, in order:

1. Loads the image.
2. Extracts 9 content features from a 4x-downsampled copy of it.
3. Runs the classifier, which answers "simple" or "complex".
4. Picks **2 cores** for simple, **4 cores** for complex.
5. Cuts the image into horizontal strips, encodes them in parallel, and
   joins them into a single normal JPEG file.

```bash
./build/jcadaptive-static <input-image> <output.jpg> [-quality Q] [-threads N]
```

- Input formats: 8-bit JPEG, BMP, PNG, PPM/PGM.
- `-quality Q` — JPEG quality, 0-100 (default 85).
- `-threads N` — ignore the model's decision and force this core count.
  Useful for comparing against the adaptive choice.

Example:

```
$ ./build/jcadaptive-static testimages/DIV2K_valid_HR/0802.png out.jpg
testimages/DIV2K_valid_HR/0802.png: 2040x1356, predicted complex (P(simple)=0.020 P(complex)=0.980) -> 4 cores
out.jpg: wrote 494971 bytes (4 strips spliced, 4 cores)
```

The first line shows the model's prediction and the core count chosen; the
second confirms the output file.

## `jcadaptivebench-static` — measurement tool

Source: [src/jcadaptivebench.c](src/jcadaptivebench.c)

Used to produce the project's evaluation numbers, not to encode images for
use. Over a folder of images, it runs four things per image, each repeated
for a couple of seconds and timestamped:

- the classifier alone,
- an always-2-core encode,
- an always-4-core encode,
- the adaptive pipeline (classifier, then encode at the chosen core count).

It prints a CSV of timings and writes no JPEG files. It is normally run
through the wrapper script rather than directly, which also records power
on a Jetson and analyses the results:

```bash
./scripts/measure/run_measure.sh testimages/DIV2K_valid_HR
```

This gives the **energy per image** for the three policies and the
**classifier overhead** as a share of encode time. See
[scripts/measure/README.md](scripts/measure/README.md). Linux and macOS
only.

## `jcparallelbench-static` — fixed-core timing benchmark

Source: [src/jcparallelbench.c](src/jcparallelbench.c)

Times the parallel encoder alone, at one core count you choose, over a
folder of images. It does not involve the model. Its output is what the
model is trained from: run once with 2 cores and once with 4, and the two
timing tables become the training labels.

```bash
./build/jcparallelbench-static <dataset_dir> --threads N [--quality Q] [--repeat R]
```

Normally run through `scripts/bench_2vs4.sh`, which does both runs and
merges them. See [scripts/ml/README.md](scripts/ml/README.md) for how the
results feed the training.

## `jcparalleltest-static` — correctness test

Source: [src/jcparalleltest.c](src/jcparalleltest.c)

Confirms the parallel encoder is correct. It generates a test image,
encodes it the normal single-threaded way as a reference, then encodes it
with 1, 2, 3 and 4 strips and checks that the decoded pixels match the
reference exactly. It takes no arguments.

```bash
./build/jcparalleltest-static
```

The last line should read `ALL PASS`. Run this once after every build,
before trusting any other output.

## `cjpeg-static` and `djpeg-static` — stock reference tools

These are the standard libjpeg-turbo command-line encoder and decoder,
unchanged by this project. They are useful as a reference: for example, to
show that the adaptive encoder's output decodes to the same pixels as a
normal single-threaded encode.

```bash
./build/cjpeg-static -quality 85 -outfile ref.jpg input.png
./build/djpeg-static -outfile ref.ppm ref.jpg
./build/djpeg-static -outfile adaptive.ppm out.jpg
cmp ref.ppm adaptive.ppm && echo "IDENTICAL PIXELS"
```

---

## Which one do I run?

| I want to... | Run |
|---|---|
| Encode an image with the adaptive pipeline | `jcadaptive-static` |
| Check the build is working correctly | `jcparalleltest-static` |
| Get energy and overhead numbers for the report | `scripts/measure/run_measure.sh` (uses `jcadaptivebench-static`) |
| Collect 2-core vs 4-core timings to retrain the model | `scripts/bench_2vs4.sh` (uses `jcparallelbench-static`) |
| Compare against a normal single-threaded encode | `cjpeg-static` and `djpeg-static` |
