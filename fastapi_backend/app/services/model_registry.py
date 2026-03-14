from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

try:
    from tensorflow.keras.models import load_model
except Exception:  # pragma: no cover
    load_model = None

from .preprocessing import get_project_root


@dataclass
class LoadedModels:
    brain_model: Any | None
    liver_rf_model: Any | None
    liver_xgb_model: Any | None
    liver_lgbm_model: Any | None
    liver_scaler: Any | None
    liver_label_encoder: Any | None
    health_prediction_model: Any | None
    kidney_model: Any | None
    meta_model: Any | None


class ModelRegistry:
    def __init__(self) -> None:
        self.model_dir = get_project_root() / "model"
        self.models = LoadedModels(
            brain_model=self._load_brain_model(),
            liver_rf_model=self._load_joblib("liver_rf_model.joblib"),
            liver_xgb_model=self._load_joblib("liver_xgb_model.joblib"),
            liver_lgbm_model=self._load_joblib("liver_lgbm_model.joblib"),
            liver_scaler=self._load_joblib("liver_scaler.joblib"),
            liver_label_encoder=self._load_joblib("liver_label_encoder.joblib"),
            health_prediction_model=self._load_pickle("health_prediction_model.pkl"),
            kidney_model=self._load_pickle("kidney_model.pkl"),
            meta_model=self._load_joblib("meta_model.joblib"),
        )

    def _path(self, filename: str) -> Path:
        return self.model_dir / filename

    def _load_joblib(self, filename: str) -> Any | None:
        path = self._path(filename)
        if not path.exists():
            return None
        try:
            payload = joblib.load(path)
            # New meta-model artifact format stores model + metadata in a dict.
            if filename == "meta_model.joblib" and isinstance(payload, dict) and "meta_model" in payload:
                return payload["meta_model"]
            return payload
        except Exception:
            return None

    def _load_pickle(self, filename: str) -> Any | None:
        return self._load_joblib(filename)

    def _load_brain_model(self) -> Any | None:
        if load_model is None:
            return None
        candidates = ["brain_tumor_model.h5", "brain_tumor.h5"]
        for candidate in candidates:
            path = self._path(candidate)
            if not path.exists():
                continue
            try:
                return load_model(path)
            except Exception:
                continue
        return None

    def predict_binary_probability(self, model: Any, features: np.ndarray) -> float | None:
        if model is None:
            return None
        try:
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(features)
                return float(probs[0][1])
            pred = model.predict(features)
            pred_arr = np.asarray(pred).reshape(-1)
            value = float(pred_arr[0])
            return min(max(value, 0.0), 1.0)
        except Exception:
            return None


registry = ModelRegistry()
