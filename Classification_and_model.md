# How images are classified as simple or complex, and the model built from it

The adaptive encoder decides, before encoding, whether an image is
**simple** (encode on 2 cores) or **complex** (encode on 4 cores). This
document explains where those two labels come from and how the model that
predicts them was built.

In short: an image is labelled by **how long it takes to encode per
megapixel**, measured on a benchmark. The fastest 17% of images are
"simple"; the rest are "complex". A Random Forest is then trained to
predict that label from the image's content alone, so the decision can be
made without encoding the image first.

## Part 1: How the labels were made

### The dataset

The DIV2K high-resolution photo set, in its standard split:

| Split | Images | Used for |
|---|---|---|
| `DIV2K_train_HR` | 800 | training the model |
| `DIV2K_valid_HR` | 100 | checking the model on images it has not seen |

### Step 1: time every image

`jcparallelbench-static` encodes each image 50 times with 2 cores and 50
times with 4 cores, and records the mean and minimum encode time. Only the
encode is timed, not loading the image. This gives one row per image:

```
image, width, height, mean_2core_s, min_2core_s, mean_4core_s, min_4core_s, speedup_min
```

Script: `scripts/bench_2vs4.sh`. The timings used for the current model
were collected on the development Mac, not on the Jetson Nano.

### Step 2: remove the effect of image size

Raw encode time mostly reflects how many pixels an image has, not how hard
its content is. A large, flat image can take longer than a small, detailed
one. So the time is divided by the image's size:

```
time_per_mpx = mean_2core_s / (width x height / 1,000,000)
```

This is the 2-core encode time per megapixel. It measures content
difficulty independently of resolution.

### Step 3: apply a cutoff

Images are sorted by `time_per_mpx`. The lowest 17% are labelled **simple
(0)** and the rest **complex (1)**.

| Split | Simple | Complex | Cutoff (2-core time per megapixel) |
|---|---|---|---|
| Train | 137 | 663 | 0.752 ms |
| Validation | 18 | 82 | 0.754 ms |

Across the training set, `time_per_mpx` ranges from 0.59 ms to 1.13 ms,
with a median of 0.83 ms.

Script: `scripts/ml/build_dataset.py` (the 17% is its `--simple-fraction`
option).

### Points to be aware of

- **The 17% cutoff is a choice, not a measured boundary.** Nothing in the
  data marks 17% as the point where 2 cores becomes the better option.
- **The label does not say 2 cores is faster for simple images.** In the
  benchmark, 4 cores was faster than 2 for every image: the speedup ranged
  from about 1.66x to 2x. A label based on "does 4 cores help?" was
  considered and rejected for that reason, since it would have labelled
  every image complex. "Simple" therefore means "cheap to encode", on the
  reasoning that a cheap image is where giving up 2 cores costs the least.
- **The cutoff is computed separately for each split**, as the 17th
  percentile of that split, which is why the two cutoffs differ slightly.
- **The labels are tied to the machine they were timed on.** Retraining for
  the Jetson Nano means re-running the benchmark there.

## Part 2: The model

### What it has to do

Predict the simple/complex label from the image's pixels, before encoding,
and cheaply enough that the prediction does not cost more than the encode
it is trying to save.

### Input: 9 features

Each image is reduced to 9 numbers. All are averages per pixel or per
block, so none of them reveals the image's size.

| Feature | What it measures |
|---|---|
| `bytes_per_mpx` | Source file size per megapixel |
| `entropy` | Spread of brightness values (Shannon entropy of the grayscale histogram) |
| `edge_density` | Average edge strength (mean Sobel gradient) |
| `edge_variance` | How much edge strength varies across the image |
| `r_variance`, `g_variance`, `b_variance` | Variance of each colour channel |
| `mean_saturation` | Average colour saturation |
| `dct_ac_energy` | Average detail per 8x8 block, using the same transform JPEG uses |

**The features are computed on a 4x-downsampled image**: every 4th pixel
of every 4th row. At full resolution, extracting the features took several
times longer than the encode itself. Downsampling by 4 cuts that cost about
16x for a small loss in accuracy. `bytes_per_mpx` is the exception; it uses
the full image size.

Typical values in the training set show how the two classes differ:

| Feature | Simple (median) | Complex (median) |
|---|---|---|
| `dct_ac_energy` | 446 | 942 |
| `edge_density` | 51 | 103 |
| `edge_variance` | 7,039 | 13,887 |
| `bytes_per_mpx` | 1.21 million | 1.64 million |
| `entropy` | 7.14 | 7.54 |

Complex images have roughly twice the edge strength and block detail of
simple ones. The colour features differ much less.

Scripts: `scripts/ml/extract_features.py` (training) and
`src/jcfeatures.c` (the C version the encoder uses). The two agree to about
1 part in a million.

### The model: a Random Forest

A Random Forest is a set of decision trees. Each tree asks a series of
yes/no questions about the features (for example, "is `dct_ac_energy`
below 600?") and votes simple or complex; the forest's answer is the
average of the votes, given as two probabilities.

| Setting | Value |
|---|---|
| Trees | 50 |
| Maximum depth | 8 |
| Total decision nodes | 3,572 |
| Class weighting | balanced |
| Library | scikit-learn `RandomForestClassifier` |

Why these choices:

- **A Random Forest rather than a neural network**, because there are only
  900 images and 9 hand-made features. A deep network would overfit and
  would be costly to run on a small board.
- **Balanced class weighting**, because 83% of images are complex. Without
  it, a model can score 83% accuracy by always answering "complex".
- **A small forest**, because larger and deeper ones did not improve
  validation results, and a small one exports to compact C code.

Script: `scripts/ml/train_model.py`.

### How well it works

On the 100 validation images:

| | Predicted simple | Predicted complex |
|---|---|---|
| **Actually simple (18)** | 15 | 3 |
| **Actually complex (82)** | 4 | 78 |

- Accuracy: 93%
- ROC-AUC: 0.958
- Simple images found: 15 of 18 (83%)

Accuracy alone is misleading here, since always answering "complex" would
already score 82%. The number to watch is how many simple images are
found. With only 18 simple images in validation, each one is worth about
5.5 percentage points, so small differences between model versions are
within noise.

Which features the model relies on most:

| Feature | Importance |
|---|---|
| `dct_ac_energy` | 33% |
| `bytes_per_mpx` | 27% |
| `edge_density` | 18% |
| `edge_variance` | 8% |
| The other five combined | 14% |

The model mostly looks at texture and detail, which is what makes JPEG
encoding slower.

### How it runs inside the encoder

The trained model is converted to plain C by `m2cgen`
(`scripts/ml/export_c_model.py`), producing `src/complexity_model.c`: one
function, `score()`, made of nested `if` statements, with no library
dependencies. No Python is needed on the device. One prediction takes
under a microsecond; nearly all of the classifier's cost is the feature
extraction (about 0.74 ms per 2-megapixel image on the development Mac).

`jcadaptive-static` calls the feature extractor, then `score()`, and picks
2 cores if "simple" is the more likely class, otherwise 4. The C encoder's
predictions matched the Python model on all 100 validation images.

## The whole process in order

| Step | Script or program | Output |
|---|---|---|
| 1. Time every image at 2 and 4 cores | `scripts/bench_2vs4.sh` | `bench_results/dataset_*.csv` |
| 2. Label by time per megapixel | `scripts/ml/build_dataset.py` | `bench_results/labels_*.csv` |
| 3. Extract the 9 features | `scripts/ml/extract_features.py` | `bench_results/features_*.csv` |
| 4. Train and evaluate the forest | `scripts/ml/train_model.py` | `bench_results/complexity_model.joblib` |
| 5. Export to C | `scripts/ml/export_c_model.py` | `src/complexity_model.c` |
| 6. Use it | `build/jcadaptive-static` | the encoded JPEG |

Full commands for each step are in
[scripts/ml/README.md](scripts/ml/README.md).
