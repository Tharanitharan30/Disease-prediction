from __future__ import annotations

from typing import Any

import numpy as np


META_FEATURES = [
    "brain_tumor_prob",
    "liver_rf_prob",
    "liver_xgb_prob",
    "liver_lgbm_prob",
    "health_model_prob",
    "kidney_model_prob",
]


def normalize_scores(raw_scores: dict[str, float | None]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key in META_FEATURES:
        value = raw_scores.get(key)
        normalized[key] = 0.0 if value is None else float(min(max(value, 0.0), 1.0))
    return normalized


def run_meta_model(meta_model: Any | None, specialist_scores: dict[str, float]) -> tuple[str, float, str]:
    features = np.array([[specialist_scores[key] for key in META_FEATURES]], dtype=np.float32)

    if meta_model is not None:
        try:
            if hasattr(meta_model, "predict_proba"):
                prob = float(meta_model.predict_proba(features)[0][1])
            else:
                pred = float(np.asarray(meta_model.predict(features)).reshape(-1)[0])
                prob = min(max(pred, 0.0), 1.0)
            label = "Tumor Detected" if prob >= 0.5 else "No Tumor"
            return label, prob, "Meta Stacking Model"
        except Exception:
            pass

    # Fallback weighted voting when a trained meta model is not available yet.
    weights = {
        "brain_tumor_prob": 0.45,
        "liver_rf_prob": 0.15,
        "liver_xgb_prob": 0.15,
        "liver_lgbm_prob": 0.15,
        "health_model_prob": 0.05,
        "kidney_model_prob": 0.05,
    }
    score = sum(specialist_scores[key] * weights[key] for key in weights)
    label = "Tumor Detected" if score >= 0.5 else "No Tumor"
    return label, float(score), "Weighted Ensemble (fallback)"
