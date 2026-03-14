from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import Body, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator
from PIL import Image

try:
    from tensorflow.keras.models import load_model
except Exception:  # pragma: no cover
    load_model = None


app = FastAPI(title="Medical AI Diagnosis API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)


MODEL_STORE: dict[str, Any] = {}


class KidneyPayload(BaseModel):
    features: list[float] = Field(..., description="24 tabular kidney features")

    @field_validator("features")
    @classmethod
    def validate_len(cls, value: list[float]) -> list[float]:
        if len(value) != 24:
            raise ValueError("Kidney endpoint expects exactly 24 features")
        return value


class HealthPayload(BaseModel):
    features: list[float]


def _extract_features_from_any(payload: Any) -> list[float]:
    if isinstance(payload, list):
        return [float(v) for v in payload]
    if isinstance(payload, dict) and isinstance(payload.get("features"), list):
        return [float(v) for v in payload["features"]]
    raise HTTPException(status_code=400, detail="Expected JSON list or {'features': [...]} payload")


class LiverPayload(BaseModel):
    age: float
    gender: str
    total_bilirubin: float
    direct_bilirubin: float
    alkaline_phosphotase: float
    alamine_aminotransferase: float
    aspartate_aminotransferase: float
    total_proteins: float
    albumin: float
    albumin_and_globulin_ratio: float


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _model_dir() -> Path:
    return _project_root() / "model"


def _risk_level(confidence: float) -> str:
    if confidence >= 75:
        return "High"
    if confidence >= 45:
        return "Medium"
    return "Low"


# Core helper that fixes the common 0% confidence bug.
def get_proba(model: Any, X: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        probs = np.asarray(probs)
        if probs.ndim == 1:
            return float(probs[0])
        if probs.shape[1] == 1:
            return float(probs[0, 0])
        return float(probs[0, 1])

    raw = model.predict(X)
    raw = np.asarray(raw)
    if raw.ndim == 2 and raw.shape[1] == 1:
        return float(raw[0, 0])
    if raw.ndim == 2 and raw.shape[1] >= 2:
        return float(raw[0, 1])
    if raw.ndim == 1:
        return float(raw[0])
    return float(raw.reshape(-1)[0])


def _load_joblib(filename: str) -> Any | None:
    path = _model_dir() / filename
    if not path.exists():
        return None
    try:
        payload = joblib.load(path)
        if filename == "meta_model.joblib" and isinstance(payload, dict) and "meta_model" in payload:
            return payload["meta_model"]
        return payload
    except Exception:
        return None


def _load_keras_model(candidates: list[str]) -> Any | None:
    if load_model is None:
        return None
    for filename in candidates:
        path = _model_dir() / filename
        if not path.exists():
            continue
        try:
            return load_model(path)
        except Exception:
            continue
    return None


def _preprocess_image(content: bytes) -> np.ndarray:
    image = Image.open(io.BytesIO(content)).convert("RGB")
    image = image.resize((224, 224))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def _liver_vector_from_10_features(payload: LiverPayload) -> np.ndarray:
    gender_num = 1.0 if payload.gender.strip().lower() in {"m", "male", "1"} else 0.0

    # Map 10 user inputs into the 15 feature schema expected by the trained liver pipeline.
    features_15 = np.array(
        [
            payload.total_bilirubin,  # Liver_Function_Test (proxy)
            payload.aspartate_aminotransferase,  # AST
            1.0 if payload.albumin < 3.5 else 0.0,  # Sym_Fatigue (derived)
            1.0 if payload.direct_bilirubin > 0.3 else 0.0,  # Sym_Dark_Urine (derived)
            payload.alamine_aminotransferase,  # ALT
            1.0 if payload.age > 55 else 0.0,  # Comorb_Diabetes (derived proxy)
            payload.albumin,  # Albumin
            payload.total_bilirubin,  # Bilirubin
            1.0 if payload.total_bilirubin > 1.2 else 0.0,  # Sym_Abdominal_Pain (derived)
            max(120.0, payload.total_proteins * 35.0),  # Platelets (derived proxy)
            payload.alkaline_phosphotase,  # Alk_Phosphatase
            1.0 if payload.direct_bilirubin > 0.4 else 0.0,  # Sym_Itching (derived)
            1.0 if payload.albumin_and_globulin_ratio < 1.0 else 0.0,  # Sym_Ascites (derived)
            1.0 if payload.total_proteins < 6.2 else 0.0,  # Sym_Weight_Loss (derived)
            1.0 if payload.total_bilirubin > 1.1 else 0.0,  # Sym_Jaundice (derived)
        ],
        dtype=np.float32,
    )

    # Inject small gender effect into first feature to keep signal in model path.
    features_15[0] = float(features_15[0] + 0.02 * gender_num)
    return features_15.reshape(1, -1)


def _binary_response(
    organ: str,
    model_name: str,
    prob: float,
    positive_label: str,
    negative_label: str,
    breakdown: dict[str, float] | None = None,
) -> dict[str, Any]:
    prob = float(np.clip(prob, 0.0, 1.0))
    confidence = round(prob * 100.0, 2)
    prediction = positive_label if prob >= 0.5 else negative_label
    payload: dict[str, Any] = {
        "organ": organ,
        "model": model_name,
        "prediction": prediction,
        "confidence": confidence,
        "risk_level": _risk_level(confidence),
    }
    if breakdown:
        payload["breakdown"] = breakdown
    return payload


def _predict_label_and_confidence(model: Any, X: np.ndarray, label_encoder: Any | None = None) -> tuple[str, float, np.ndarray]:
    if not hasattr(model, "predict_proba"):
        prob = get_proba(model, X)
        label = "Positive" if prob >= 0.5 else "Negative"
        return label, float(np.clip(prob, 0.0, 1.0)), np.array([1 - prob, prob], dtype=np.float32)

    probs = np.asarray(model.predict_proba(X))[0]
    class_index = int(np.argmax(probs))
    confidence = float(np.clip(probs[class_index], 0.0, 1.0))
    if label_encoder is not None and hasattr(label_encoder, "classes_"):
        label = str(label_encoder.classes_[class_index])
    else:
        label = str(class_index)
    return label, confidence, probs


@app.on_event("startup")
def load_models_once() -> None:
    MODEL_STORE["brain_model_a"] = _load_keras_model(["brain_tumor_model.h5"])
    MODEL_STORE["brain_model_b"] = _load_keras_model(["brain_tumor.h5"])

    MODEL_STORE["health_model"] = _load_joblib("health_prediction_model.pkl")
    MODEL_STORE["kidney_model"] = _load_joblib("kidney_model.pkl")

    MODEL_STORE["liver_scaler"] = _load_joblib("liver_scaler.joblib")
    MODEL_STORE["liver_lgbm"] = _load_joblib("liver_lgbm_model.joblib")
    MODEL_STORE["liver_rf"] = _load_joblib("liver_rf_model.joblib")
    MODEL_STORE["liver_xgb"] = _load_joblib("liver_xgb_model.joblib")
    MODEL_STORE["liver_label_encoder"] = _load_joblib("liver_label_encoder.joblib")

    MODEL_STORE["meta_model"] = _load_joblib("meta_model.joblib")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict/brain")
async def predict_brain(file: UploadFile = File(...)) -> dict[str, Any]:
    model_a = MODEL_STORE.get("brain_model_a")
    model_b = MODEL_STORE.get("brain_model_b")
    if model_a is None and model_b is None:
        raise HTTPException(status_code=500, detail="Brain models are not loaded")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")

    X = _preprocess_image(content)

    probs: list[float] = []
    breakdown: dict[str, float] = {}
    if model_a is not None:
        p1 = float(np.clip(get_proba(model_a, X), 0.0, 1.0))
        probs.append(p1)
        breakdown["brain_tumor_model.h5"] = round(p1 * 100.0, 2)
    if model_b is not None:
        p2 = float(np.clip(get_proba(model_b, X), 0.0, 1.0))
        probs.append(p2)
        breakdown["brain_tumor.h5"] = round(p2 * 100.0, 2)

    ensemble_prob = float(np.mean(probs))
    return _binary_response(
        organ="Brain",
        model_name="Brain CNN Ensemble",
        prob=ensemble_prob,
        positive_label="Tumor Detected",
        negative_label="No Tumor",
        breakdown=breakdown,
    )


@app.post("/predict/kidney")
def predict_kidney(payload: Any = Body(...)) -> dict[str, Any]:
    model = MODEL_STORE.get("kidney_model")
    if model is None:
        raise HTTPException(status_code=500, detail="Kidney model is not loaded")

    features = _extract_features_from_any(payload)
    if len(features) != 24:
        raise HTTPException(status_code=400, detail="Kidney endpoint expects exactly 24 features")

    X = np.array(features, dtype=np.float32).reshape(1, -1)
    prob = float(np.clip(get_proba(model, X), 0.0, 1.0))
    return _binary_response(
        organ="Kidney",
        model_name="Kidney Disease Classifier",
        prob=prob,
        positive_label="Kidney Disease Detected",
        negative_label="No Kidney Disease",
        breakdown={"kidney_model.pkl": round(prob * 100.0, 2)},
    )


@app.post("/predict/health")
def predict_health(payload: Any = Body(...)) -> dict[str, Any]:
    model = MODEL_STORE.get("health_model")
    if model is None:
        raise HTTPException(status_code=500, detail="Health model is not loaded")
    features = _extract_features_from_any(payload)
    if not features:
        raise HTTPException(status_code=400, detail="features list cannot be empty")

    X = np.array(features, dtype=np.float32).reshape(1, -1)
    prob = float(np.clip(get_proba(model, X), 0.0, 1.0))
    return _binary_response(
        organ="Health",
        model_name="General Health Classifier",
        prob=prob,
        positive_label="Health Risk Detected",
        negative_label="Low Health Risk",
        breakdown={"health_prediction_model.pkl": round(prob * 100.0, 2)},
    )


@app.post("/predict/liver")
def predict_liver(payload: LiverPayload) -> dict[str, Any]:
    scaler = MODEL_STORE.get("liver_scaler")
    lgbm = MODEL_STORE.get("liver_lgbm")
    rf = MODEL_STORE.get("liver_rf")
    xgb = MODEL_STORE.get("liver_xgb")
    label_encoder = MODEL_STORE.get("liver_label_encoder")

    if lgbm is None and rf is None and xgb is None:
        raise HTTPException(status_code=500, detail="Liver models are not loaded")

    X_raw = _liver_vector_from_10_features(payload)
    X = scaler.transform(X_raw) if scaler is not None else X_raw

    model_probs: dict[str, np.ndarray] = {}
    model_confidences: dict[str, float] = {}
    model_labels: dict[str, str] = {}

    for model_name, model_obj in [
        ("liver_lgbm_model.joblib", lgbm),
        ("liver_rf_model.joblib", rf),
        ("liver_xgb_model.joblib", xgb),
    ]:
        if model_obj is None:
            continue
        label, conf, probs = _predict_label_and_confidence(model_obj, X, label_encoder=label_encoder)
        model_probs[model_name] = probs
        model_confidences[model_name] = round(conf * 100.0, 2)
        model_labels[model_name] = label

    if not model_probs:
        raise HTTPException(status_code=500, detail="No liver model produced probabilities")

    prob_vectors = list(model_probs.values())
    min_len = min(vec.shape[0] for vec in prob_vectors)
    aligned = [vec[:min_len] for vec in prob_vectors]
    ensemble_probs = np.mean(np.stack(aligned, axis=0), axis=0)

    ensemble_idx = int(np.argmax(ensemble_probs))
    ensemble_conf = float(np.clip(ensemble_probs[ensemble_idx], 0.0, 1.0))

    if label_encoder is not None and hasattr(label_encoder, "classes_") and len(label_encoder.classes_) > ensemble_idx:
        prediction_label = str(label_encoder.classes_[ensemble_idx])
    else:
        prediction_label = f"Class {ensemble_idx}"

    confidence = round(ensemble_conf * 100.0, 2)
    return {
        "organ": "Liver",
        "model": "Liver Ensemble (LGBM + RF + XGB)",
        "prediction": prediction_label,
        "confidence": confidence,
        "risk_level": _risk_level(confidence),
        "breakdown": {
            "confidences": model_confidences,
            "labels": model_labels,
            "ensemble": round(confidence, 2),
        },
    }
