from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
    average_precision_score,
)


def metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    p_bin, r_bin, f1_bin, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "precision_binary": float(p_bin),
        "recall_binary": float(r_bin),
        "f1_binary": float(f1_bin),
    }


def ranking_metrics(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }


def find_best_threshold_macro_f1(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold_min: float = 0.1,
    threshold_max: float = 0.9,
    num_thresholds: int = 50,
) -> tuple[float, float]:
    thresholds = np.linspace(threshold_min, threshold_max, num_thresholds)
    f1_values = []
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        f1_values.append(f1_score(y_true, y_pred, average="macro"))
    best_idx = int(np.argmax(f1_values))
    return float(thresholds[best_idx]), float(f1_values[best_idx])
