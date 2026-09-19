"""Compute pre-encode image-content features for the complexity classifier.

Only features knowable before the JPEG encoder runs are computed here --
never anything derived from the benchmark timing columns, since those are
the label source (see build_dataset.py), not model input.

width/height are recorded for reference/joining only. The label is already
normalized by resolution (time per megapixel), so the feature set below is
kept scale-invariant (per-pixel or per-block) on purpose -- otherwise the
model could just re-derive image size instead of learning content
difficulty.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.fftpack import dct
from scipy.ndimage import sobel

FEATURE_NAMES = [
    "bytes_per_mpx",
    "entropy",
    "edge_density",
    "edge_variance",
    "r_variance",
    "g_variance",
    "b_variance",
    "mean_saturation",
    "dct_ac_energy",
]


def shannon_entropy(gray: np.ndarray) -> float:
    hist, _ = np.histogram(gray, bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist)))


def edge_stats(gray: np.ndarray) -> tuple[float, float]:
    gx = sobel(gray.astype(np.float64), axis=1)
    gy = sobel(gray.astype(np.float64), axis=0)
    magnitude = np.hypot(gx, gy)
    return float(np.mean(magnitude)), float(np.var(magnitude))


def dct_ac_energy(gray: np.ndarray) -> float:
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    blocks = gray[:h8, :w8].astype(np.float64).reshape(h8 // 8, 8, w8 // 8, 8)
    blocks = blocks.transpose(0, 2, 1, 3).reshape(-1, 8, 8)
    coeffs = dct(dct(blocks, axis=1, norm="ortho"), axis=2, norm="ortho")
    ac_energy = np.abs(coeffs).sum(axis=(1, 2)) - np.abs(coeffs[:, 0, 0])
    return float(np.mean(ac_energy))


def extract(image_path: Path) -> dict:
    with Image.open(image_path) as img:
        rgb = np.asarray(img.convert("RGB"))
        hsv = np.asarray(img.convert("HSV"))

    gray = np.asarray(Image.fromarray(rgb).convert("L"))
    megapixels = (rgb.shape[0] * rgb.shape[1]) / 1_000_000
    edge_density, edge_variance = edge_stats(gray)

    return {
        "bytes_per_mpx": image_path.stat().st_size / megapixels,
        "entropy": shannon_entropy(gray),
        "edge_density": edge_density,
        "edge_variance": edge_variance,
        "r_variance": float(np.var(rgb[:, :, 0])),
        "g_variance": float(np.var(rgb[:, :, 1])),
        "b_variance": float(np.var(rgb[:, :, 2])),
        "mean_saturation": float(np.mean(hsv[:, :, 1])),
        "dct_ac_energy": dct_ac_energy(gray),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", type=Path, default=Path("../../bench_results"))
    parser.add_argument("--testimages-dir", type=Path, default=Path("../../testimages"))
    args = parser.parse_args()

    splits = {
        "train": args.testimages_dir / "DIV2K_train_HR",
        "val": args.testimages_dir / "DIV2K_valid_HR",
    }

    for split, images_dir in splits.items():
        labels_path = args.bench_dir / f"labels_{split}.csv"
        with labels_path.open(newline="") as f:
            rows = list(csv.DictReader(f))

        out_path = args.bench_dir / f"features_{split}.csv"
        fieldnames = ["image", "width", "height", *FEATURE_NAMES]
        with out_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for i, row in enumerate(rows, 1):
                image_path = images_dir / row["image"]
                features = extract(image_path)
                writer.writerow(
                    {
                        "image": row["image"],
                        "width": row["width"],
                        "height": row["height"],
                        **features,
                    }
                )
                if i % 100 == 0 or i == len(rows):
                    print(f"{split}: {i}/{len(rows)}")

        print(f"{split}: wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
