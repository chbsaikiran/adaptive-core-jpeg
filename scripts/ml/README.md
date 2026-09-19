# Complexity classifier

Predicts whether an image is **simple** (label `0`, fast to JPEG-encode) or
**complex** (label `1`, slow to JPEG-encode), from features computable
*before* encoding. This is meant to feed the adaptive-core logic: predict
complexity up front, then pick 2 vs 4 encoder cores accordingly.

Run with [uv](https://docs.astral.sh/uv/): `uv run build_dataset.py`, etc.
Dependencies are pinned in `pyproject.toml` / `uv.lock`.

## 0. How to run

All commands below are run from this directory (`scripts/ml/`). `uv run`
creates/uses the project's `.venv` and installs dependencies automatically,
so an explicit `uv venv` + activate step isn't required — but both are
shown in case you want an activated shell instead.

```bash
cd scripts/ml

# Create the virtualenv and install pinned dependencies (numpy, pillow,
# scipy, pandas, scikit-learn, joblib) from pyproject.toml / uv.lock.
uv venv
uv sync
# (optional) activate it directly instead of prefixing every command with
# `uv run`:
source .venv/bin/activate

# 1. Derive simple/complex labels from bench_results/dataset_{train,val}.csv
uv run build_dataset.py
#    -> bench_results/labels_train.csv, labels_val.csv

# 2. Extract pre-encode image features for every labeled image
#    (reads testimages/DIV2K_train_HR and DIV2K_valid_HR — ~900 images,
#    takes a couple of minutes)
uv run extract_features.py
#    -> bench_results/features_train.csv, features_val.csv

# 3. Train the model, evaluate on the val split
uv run train_model.py
#    -> bench_results/complexity_model.joblib

# 4. Export the trained model to standalone C, for on-device inference
#    (e.g. Jetson Nano) with no Python/runtime dependency -- see "Deploying
#    to Jetson Nano" below
uv run export_c_model.py
#    -> bench_results/complexity_model.c
```

This requires `bench_results/dataset_train.csv` and `dataset_val.csv` to
already exist (produced by `scripts/run_bench.sh` + `scripts/merge_bench.py`
— see the repo root `Steps_to_build_and_benchmark_*.md`), and
`testimages/DIV2K_train_HR` / `DIV2K_valid_HR` to be present locally (both
are gitignored — see the note in `.gitignore`).

Re-running `build_dataset.py` with a different split, e.g.
`uv run build_dataset.py --simple-fraction 0.2`, changes the simple/complex
cutoff — re-run `extract_features.py` and `train_model.py` after (features
don't change, but the label file they join against does, so `train_model.py`
alone is actually sufficient unless you also moved `--bench-dir`).

## 1. Where the data comes from

`bench_results/dataset_train.csv` (800 images, DIV2K_train_HR) and
`dataset_val.csv` (100 images, DIV2K_valid_HR) come from the existing
`jcparallelbench` benchmark pipeline (`scripts/run_bench.sh` +
`scripts/merge_bench.py`). Each row has:

```
image,width,height,mean_2core_s,min_2core_s,mean_4core_s,min_4core_s,speedup_min
```

There was no label column and no image-content feature extraction in the
repo before this — both had to be built.

## 2. Label derivation (`build_dataset.py`)

Naively thresholding raw `mean_2core_s` doesn't work: encode time is
dominated by *resolution*, not content difficulty. A big, flat/simple image
can take longer to encode than a small, busy one purely because it has more
pixels. So the script normalizes first:

```
time_per_mpx = mean_2core_s / (width * height / 1_000_000)
```

and thresholds *that* — the bottom ~17% of images by time-per-megapixel are
labeled `simple (0)`, the rest `complex (1)`. The `--simple-fraction` flag
controls the split (default `0.17`, chosen to match the ~139/669 ratio).

`speedup_min` (`min_2core_s / min_4core_s`) was considered as the label
basis instead — the docs in `Steps_to_build_and_benchmark_*.md` sketch a
">1 = complex" rule — but rejected: in the actual benchmark data it never
drops below ~1.66, so that rule would label *every* image complex.

Output: `bench_results/labels_train.csv`, `labels_val.csv` (input CSV plus
`time_per_mpx` and `label` columns). Actual result on this dataset:

| split | simple | complex |
|---|---|---|
| train | 137 | 663 |
| val | 18 | 82 |

## 3. Feature extraction (`extract_features.py`)

For each image, loaded with Pillow, computes 9 features — all scale-invariant
(per-pixel or per-block averages), so none of them can just re-encode image
size, which matters since the label itself is already resolution-normalized:

| feature | what it captures |
|---|---|
| `bytes_per_mpx` | PNG file size per megapixel — cheap complexity proxy |
| `entropy` | grayscale pixel histogram Shannon entropy |
| `edge_density` | mean Sobel gradient magnitude |
| `edge_variance` | variance of Sobel gradient magnitude (texture proxy) |
| `r_variance`, `g_variance`, `b_variance` | per-channel pixel variance |
| `mean_saturation` | mean HSV saturation |
| `dct_ac_energy` | mean absolute AC energy of 8x8-block DCTs — the closest proxy to what the JPEG encoder itself spends time on |

`width`/`height` are also recorded in `features_*.csv` for reference/joining
but are deliberately **not** used as model input features (see above).

## 4. Model (`train_model.py`)

Given the small size (900 images total) and tabular/engineered features
(not raw pixels), a deep CNN would be overkill and prone to overfitting. A
class-balanced `sklearn.ensemble.RandomForestClassifier`
(`class_weight="balanced"`) is trained on the existing train/val split (no
re-splitting), with a deliberately modest `n_estimators=50, max_depth=8`
(configurable via CLI flags) — deeper/larger forests were tried and didn't
improve val performance (see below), and keeping the forest small matters
for step 6 (C export). Random Forest specifically (over e.g.
`HistGradientBoostingClassifier`, which scored similarly) because it's the
one `export_c_model.py` can turn into plain C via `m2cgen` — m2cgen doesn't
support sklearn's histogram-based boosting models.

Saved to `bench_results/complexity_model.joblib` (a dict with `model` and
`feature_names`).

**Result on this dataset:** val ROC-AUC **0.983**.

```
              precision    recall  f1-score   support
      simple       0.89      0.89      0.89        18
     complex       0.98      0.98      0.98        82
```

Top features by importance: `dct_ac_energy`, `edge_density`,
`bytes_per_mpx` — i.e. the model is mostly keying off texture/detail, which
lines up with what actually makes JPEG encoding slower.

## 5. How to judge whether the model is actually good (imbalance caveat)

The dataset is skewed ~83% complex / 17% simple, so **accuracy is
misleading on its own** — a model that always predicts "complex" would
still score ~83% accuracy while being useless. Check these instead:

- **Recall on the `simple` class specifically.** This is the number that
  tells you whether the model can actually find the minority class, which
  is the whole point. `train_model.py`'s classification report prints this
  per class — look at the `simple` row, not the overall `accuracy` line.
- **Confusion matrix**, not just aggregate metrics — shows exactly how many
  simple images get misclassified as complex (false negatives) vs. the
  reverse, so you can see which error the model is actually making.
- **Macro-averaged F1** (unweighted mean across both classes) over the
  weighted/overall F1 — the weighted average is dominated by the majority
  class and can look good even when the minority class is being ignored.
- **ROC-AUC**, which is threshold-independent and less distorted by class
  imbalance than accuracy, but still worth pairing with the metrics above
  since it can look fine even when the default 0.5 decision threshold
  performs poorly on the minority class.
- **Compare against a trivial baseline** — a "always predict complex"
  classifier gets ~82% accuracy on this val set purely from the class
  ratio; anything you report should be read relative to that baseline, not
  in isolation.
- **Watch for train/val gap.** With only 137 simple training examples,
  overfitting is a real risk — if train metrics are near-perfect but val
  recall on `simple` is much lower, that's a sign to simplify the model
  (fewer trees / shallower depth) or gather more simple-labeled images
  rather than trust the number as-is.
- Because the val set only has 18 simple images, individual misclassifications
  swing the recall number a lot (each one is ~5.5 percentage points) — treat
  differences smaller than that as noise, and re-run with a different
  `--simple-fraction` split or more data before concluding one model
  variant is meaningfully better than another.

## 6. Deploying to Jetson Nano (`export_c_model.py`)

The joblib/pickle file (`complexity_model.joblib`) is **not** what should
ship to the Nano: it needs a full Python + numpy/scipy/scikit-learn stack
on-device (heavy for a 2–4GB board), and pickle is fragile across
sklearn/numpy/scipy version mismatches between the training machine and the
device.

Instead, `export_c_model.py` uses [`m2cgen`](https://github.com/BayesWitnesses/m2cgen)
to convert the trained forest into a standalone `score(input, output)` C
function with **no runtime dependencies** (just `<string.h>`), so it can be
compiled directly into `jcparallelbench` or any other C caller:

```bash
uv run export_c_model.py
#    -> bench_results/complexity_model.c  (~50 trees, ~3200 nodes, ~300 KB)
```

The generated file's header comment documents the calling convention:

```c
void score(double * input, double * output);
// input[0..8]  = the 9 features, in FEATURE_NAMES order (see extract_features.py)
// output[0]    = P(simple), output[1] = P(complex)
// predicted label = argmax(output)   // 0 = simple, 1 = complex
```

This was verified by cross-checking the C output against
`model.predict_proba()` in Python on several val images — probabilities
matched to 6 decimal places, and the file compiles cleanly with plain
`gcc -O2` (no external libraries beyond `-lm`).

**Caveat:** this only exports the *classifier*. `extract_features.py` (the
9 pre-encode features) is currently Python/Pillow/scipy and still needs a C
reimplementation (Sobel gradients, block DCT, histogram entropy, etc. are
all straightforward in C, but that port hasn't been done yet) before the
whole pipeline can run on the Nano without Python.

Re-run `export_c_model.py` any time `train_model.py` produces a new
`complexity_model.joblib` — the generated C file is derived output, not
meant to be hand-edited.
