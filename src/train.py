from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from data import (
    BASELINE_FEATURES,
    TARGET_COLUMN,
    load_dreaddit_dataframe,
    save_splits,
    stratified_train_val_test_split,
    validate_feature_columns,
)
from eval import find_best_threshold_macro_f1, metrics_from_predictions, ranking_metrics
from explain import save_logistic_coefficients


def train_baseline(
    random_state: int = 42,
    output_dir: Path | str = "reports/results",
    split_dir: Path | str = "data",
) -> dict[str, float]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    df = load_dreaddit_dataframe(prefer_local_raw=True)
    validate_feature_columns(df, BASELINE_FEATURES + [TARGET_COLUMN])

    X = df[BASELINE_FEATURES]
    y = df[TARGET_COLUMN].astype(int)

    split = stratified_train_val_test_split(y=y, random_state=random_state)
    save_splits(split, split_dir)

    X_train, y_train = X.loc[split.train], y.loc[split.train]
    X_val, y_val = X.loc[split.val], y.loc[split.val]
    X_test, y_test = X.loc[split.test], y.loc[split.test]

    pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=random_state)),
        ]
    )
    pipeline.fit(X_train, y_train)

    val_proba = pipeline.predict_proba(X_val)[:, 1]
    best_threshold, best_val_macro_f1 = find_best_threshold_macro_f1(
        y_true=y_val.to_numpy(), y_score=val_proba
    )

    test_proba = pipeline.predict_proba(X_test)[:, 1]
    test_pred = (test_proba >= best_threshold).astype(int)

    metrics = {}
    metrics.update(metrics_from_predictions(y_test.to_numpy(), test_pred))
    metrics.update(ranking_metrics(y_test.to_numpy(), test_proba))
    metrics["selected_threshold"] = float(best_threshold)
    metrics["val_macro_f1_at_selected_threshold"] = float(best_val_macro_f1)
    metrics["n_train"] = int(len(split.train))
    metrics["n_val"] = int(len(split.val))
    metrics["n_test"] = int(len(split.test))

    with (output_path / "baseline_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (output_path / "baseline_metrics.txt").open("w", encoding="utf-8") as f:
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

    save_logistic_coefficients(
        pipeline=pipeline,
        feature_names=BASELINE_FEATURES,
        output_csv=output_path / "baseline_logreg_coefficients.csv",
    )
    return metrics


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train baseline logistic regression with validation-based threshold selection."
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="reports/results",
        help="Directory for metrics and coefficient outputs.",
    )
    parser.add_argument(
        "--split-dir",
        type=str,
        default="data",
        help="Directory where train/val/test index .npy files are saved.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    metrics = train_baseline(
        random_state=args.seed, output_dir=args.output_dir, split_dir=args.split_dir
    )
    print("Baseline training completed.")
    for k in sorted(metrics):
        print(f"{k}: {metrics[k]}")


if __name__ == "__main__":
    main()
