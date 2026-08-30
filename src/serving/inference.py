"""Prediction helpers shared by the API and Gradio interface."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

_YES_NO = {"No": 0, "Yes": 1}
_BINARY_COLUMNS = {
    "gender": {"Female": 0, "Male": 1},
    "Partner": _YES_NO,
    "Dependents": _YES_NO,
    "PhoneService": _YES_NO,
    "PaperlessBilling": _YES_NO,
}


@lru_cache(maxsize=1)
def _load_artifacts() -> tuple[XGBClassifier, list[str]]:
    model_path = ARTIFACTS_DIR / "model.json"
    metadata_path = ARTIFACTS_DIR / "preprocessing.pkl"
    if not model_path.exists() or not metadata_path.exists():
        raise RuntimeError(
            "Model artifacts are missing. Run `uv run python scripts/run_pipeline.py "
            "--input data/raw/chum.csv` before starting the API."
        )

    metadata = joblib.load(metadata_path)
    feature_columns = metadata.get("feature_columns")
    if not isinstance(feature_columns, list) or not feature_columns:
        raise RuntimeError("Invalid preprocessing metadata: feature_columns is missing.")

    model = XGBClassifier()
    model.load_model(model_path)
    return model, feature_columns


def _prepare_features(payload: dict[str, Any], feature_columns: list[str]) -> pd.DataFrame:
    """Apply the training encodings and align the result to the saved schema."""
    frame = pd.DataFrame([payload]).copy()
    for column, mapping in _BINARY_COLUMNS.items():
        if column in frame:
            frame[column] = frame[column].map(mapping)

    categorical = frame.select_dtypes(include="object").columns.tolist()
    if categorical:
        frame = pd.get_dummies(frame, columns=categorical, dtype=int)

    frame = frame.reindex(columns=feature_columns, fill_value=0)
    return frame.astype(float)


def predict(payload: dict[str, Any]) -> dict[str, Any]:
    """Return a churn label and its model probability."""
    model, feature_columns = _load_artifacts()
    features = _prepare_features(payload, feature_columns)
    probability = float(model.predict_proba(features)[0, 1])
    return {
        "label": "Likely to churn" if probability >= 0.5 else "Not likely to churn",
        "churn_probability": round(probability, 4),
    }
