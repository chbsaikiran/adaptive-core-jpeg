"""Train and evaluate the simple/complex image classifier.

Joins the pre-encode features (extract_features.py) with the resolution-
normalized labels (build_dataset.py) and trains a class-balanced Random
Forest on the existing DIV2K train/val split, reporting metrics with a
focus on recall for the minority "simple" class -- with an ~83/17 class
split, a classifier that always predicts "complex" would score high
accuracy while being useless.

Random Forest (rather than HistGradientBoosting, which scored similarly)
is used because it's the one export_c_model.py can turn into plain C via
m2cgen for on-device (Jetson Nano) inference -- m2cgen doesn't support
sklearn's histogram-based boosting models. n_estimators/max_depth are kept
modest (50/8) since that's the config the C export ships with; deeper/more
trees didn't improve val performance here anyway (see README.md).
"""

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from extract_features import FEATURE_NAMES


def load_split(bench_dir: Path, split: str) -> pd.DataFrame:
    labels = pd.read_csv(bench_dir / f"labels_{split}.csv")[["image", "label"]]
    features = pd.read_csv(bench_dir / f"features_{split}.csv")
    return features.merge(labels, on="image", validate="one_to_one")


def evaluate(model, X_val, y_val) -> float:
    pred = model.predict(X_val)
    proba = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, proba)

    print(classification_report(y_val, pred, target_names=["simple", "complex"]))
    print("confusion matrix [rows=true, cols=pred] (simple, complex):")
    print(confusion_matrix(y_val, pred))
    print(f"ROC-AUC: {auc:.4f}")
    return auc


def print_importances(model, feature_names: list[str]) -> None:
    ranked = sorted(zip(feature_names, model.feature_importances_), key=lambda x: -x[1])
    print("\nfeature importances:")
    for feature, importance in ranked:
        print(f"  {feature:20s} {importance:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench-dir", type=Path, default=Path("../../bench_results"))
    parser.add_argument("--n-estimators", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=8)
    args = parser.parse_args()

    train = load_split(args.bench_dir, "train")
    val = load_split(args.bench_dir, "val")

    X_train, y_train = train[FEATURE_NAMES], train["label"]
    X_val, y_val = val[FEATURE_NAMES], val["label"]

    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        class_weight="balanced",
        random_state=0,
    )
    model.fit(X_train, y_train)
    auc = evaluate(model, X_val, y_val)
    print_importances(model, FEATURE_NAMES)

    out_path = args.bench_dir / "complexity_model.joblib"
    joblib.dump({"model": model, "feature_names": FEATURE_NAMES}, out_path)
    print(f"\nval ROC-AUC={auc:.4f} -> {out_path}")


if __name__ == "__main__":
    main()
