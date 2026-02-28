from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.pipeline import Pipeline


def logistic_coefficients_dataframe(
    pipeline: Pipeline, feature_names: list[str]
) -> pd.DataFrame:
    clf = pipeline.named_steps["clf"]
    coefs = clf.coef_[0]
    return (
        pd.DataFrame({"feature": feature_names, "coefficient": coefs})
        .sort_values("coefficient", ascending=False)
        .reset_index(drop=True)
    )


def save_logistic_coefficients(
    pipeline: Pipeline,
    feature_names: list[str],
    output_csv: str | Path,
) -> pd.DataFrame:
    coef_df = logistic_coefficients_dataframe(pipeline, feature_names)
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(output_path, index=False)
    return coef_df
