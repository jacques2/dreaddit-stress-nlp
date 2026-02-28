from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATASET_KAGGLE_ID = "monishakant/dataset-for-stress-analysis-in-social-media"
DATASET_CSV_NAME = "dreaddit_StressAnalysis - Sheet1.csv"

# Compact, interpretable feature set used in the baseline notebook.
BASELINE_FEATURES = [
    "syntax_ari",
    "syntax_fk_grade",
    "lex_liwc_WC",
    "lex_liwc_Authentic",
    "lex_liwc_Tone",
    "lex_dal_avg_pleasantness",
    "lex_dal_avg_activation",
    "lex_dal_avg_imagery",
    "sentiment",
    "lex_liwc_i",
    "lex_liwc_negemo",
    "lex_liwc_anx",
    "lex_liwc_sad",
    "lex_liwc_negate",
    "lex_liwc_social",
]

TARGET_COLUMN = "label"


@dataclass(frozen=True)
class SplitIndices:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def load_dreaddit_dataframe(prefer_local_raw: bool = True) -> pd.DataFrame:
    """
    Load Dreaddit data.

    Priority:
    1) local file at data/raw/dreaddit.csv
    2) kagglehub cached download
    """
    root = _repo_root()
    local_raw = root / "data" / "raw" / "dreaddit.csv"
    if prefer_local_raw and local_raw.exists():
        return pd.read_csv(local_raw)

    # Offline-friendly fallback: reuse kagglehub local cache if present.
    cache_root = Path.home() / ".cache" / "kagglehub" / "datasets"
    if cache_root.exists():
        matches = list(cache_root.rglob(DATASET_CSV_NAME))
        if matches:
            newest = max(matches, key=lambda p: p.stat().st_mtime)
            return pd.read_csv(newest)

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Could not import kagglehub and local data/raw/dreaddit.csv was not found. "
            "Install requirements or place the CSV locally."
        ) from exc

    try:
        dataset_dir = Path(kagglehub.dataset_download(DATASET_KAGGLE_ID))
    except Exception as exc:
        raise RuntimeError(
            "Could not load Dreaddit data from local file/cache and online download failed. "
            "Place data/raw/dreaddit.csv or ensure kagglehub cache is available."
        ) from exc

    csv_path = dataset_dir / DATASET_CSV_NAME
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not found at {csv_path}")
    return pd.read_csv(csv_path)


def validate_feature_columns(df: pd.DataFrame, features: Sequence[str]) -> None:
    missing = [col for col in features if col not in df.columns]
    if missing:
        raise ValueError(f"Missing expected feature columns: {missing}")


def stratified_train_val_test_split(
    y: pd.Series,
    test_size: float = 0.2,
    val_size: float = 0.1,
    random_state: int = 42,
) -> SplitIndices:
    """
    Build a three-way split from row indices.

    val_size is relative to the full dataset. With defaults:
    - train: 70%
    - val: 10%
    - test: 20%
    """
    all_indices = np.asarray(y.index)
    y_values = y.astype(int)

    train_val_idx, test_idx = train_test_split(
        all_indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y_values.loc[all_indices],
    )
    # Fraction of validation inside train_val subset.
    val_rel = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_rel,
        random_state=random_state,
        stratify=y_values.loc[train_val_idx],
    )
    return SplitIndices(
        train=np.asarray(train_idx),
        val=np.asarray(val_idx),
        test=np.asarray(test_idx),
    )


def save_splits(split: SplitIndices, output_dir: Path | str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "train_idx.npy", split.train)
    np.save(out / "val_idx.npy", split.val)
    np.save(out / "test_idx.npy", split.test)


def load_splits(input_dir: Path | str) -> SplitIndices:
    path = Path(input_dir)
    return SplitIndices(
        train=np.load(path / "train_idx.npy"),
        val=np.load(path / "val_idx.npy"),
        test=np.load(path / "test_idx.npy"),
    )
