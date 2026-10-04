"""Run the trained simple/complex classifier on one or more images.

Python counterpart of what jcadaptive-static does before it encodes:
extract the 9 pre-encode features (extract_features.py, at the same
downsample factor the model was trained at), run the Random Forest
(complexity_model.joblib, from train_model.py), and report the predicted
class and the core count the adaptive encoder would pick for it -- 2 for
simple, 4 for complex.

This only classifies; it doesn't encode anything. Use it to inspect the
model's decisions, or to cross-check the C port (src/jcfeatures.c +
src/complexity_model.c) against the Python model it was exported from --
jcadaptive-static prints the same P(simple)/P(complex) for the same image.

Each argument may be an image file or a directory of images.
"""

import argparse
import csv
import sys
from pathlib import Path

import joblib
import pandas as pd

from extract_features import DEFAULT_DOWNSAMPLE, extract

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".ppm", ".pgm"}

# Must match SIMPLE_THREADS / COMPLEX_THREADS in src/jcadaptive.c.
SIMPLE_CORES = 2
COMPLEX_CORES = 4


def collect_images(paths: list[Path]) -> list[Path]:
    images = []
    for path in paths:
        if path.is_dir():
            images.extend(
                sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            )
        else:
            images.append(path)
    return images


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("images", nargs="+", type=Path, help="image files and/or directories")
    parser.add_argument("--bench-dir", type=Path, default=Path("../../bench_results"))
    parser.add_argument(
        "--model-path", type=Path, default=None, help="defaults to <bench-dir>/complexity_model.joblib"
    )
    parser.add_argument(
        "--downsample",
        type=int,
        default=DEFAULT_DOWNSAMPLE,
        help="feature downsample factor; must be the one the model was trained at",
    )
    parser.add_argument("--features", action="store_true", help="also print the 9 feature values")
    parser.add_argument("--csv", type=Path, default=None, help="also write the results to this CSV")
    args = parser.parse_args()

    model_path = args.model_path or args.bench_dir / "complexity_model.joblib"
    bundle = joblib.load(model_path)
    model, feature_names = bundle["model"], bundle["feature_names"]

    images = collect_images(args.images)
    if not images:
        sys.exit("no images found")

    results = []
    for image_path in images:
        try:
            features = extract(image_path, args.downsample)
        except OSError as e:
            print(f"{image_path}: skipped ({e})", file=sys.stderr)
            continue

        # A one-row DataFrame, so the model sees the same column names (and
        # order) it was trained with.
        X = pd.DataFrame([features])[feature_names]
        p_simple, p_complex = model.predict_proba(X)[0]
        label = 1 if p_complex > p_simple else 0
        cores = SIMPLE_CORES if label == 0 else COMPLEX_CORES

        print(
            f"{image_path}: predicted {'simple' if label == 0 else 'complex'} "
            f"(P(simple)={p_simple:.3f} P(complex)={p_complex:.3f}) -> {cores} cores"
        )
        if args.features:
            for name in feature_names:
                print(f"    {name:16s} {features[name]:.4f}")

        results.append(
            {
                "image": image_path.name,
                "predicted": "simple" if label == 0 else "complex",
                "label": label,
                "p_simple": f"{p_simple:.4f}",
                "p_complex": f"{p_complex:.4f}",
                "cores": cores,
                **{name: features[name] for name in feature_names},
            }
        )

    if not results:
        sys.exit("no images could be read")

    if len(results) > 1:
        simple_count = sum(1 for r in results if r["label"] == 0)
        print(f"\n{len(results)} images: {simple_count} simple, {len(results) - simple_count} complex")

    if args.csv:
        with args.csv.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[*results[0].keys()])
            writer.writeheader()
            writer.writerows(results)
        print(f"wrote {len(results)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
