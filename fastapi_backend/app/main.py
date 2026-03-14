from __future__ import annotations

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .schemas import HealthResponse, PredictionResponse
from .services.meta_ensemble import normalize_scores, run_meta_model
from .services.model_registry import registry
from .services.preprocessing import (
    detect_organ_from_filename,
    image_statistics_features,
    load_rgb_image,
)

app = FastAPI(title="Disease Prediction Inference API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict/upload-scan", response_model=PredictionResponse)
async def predict_upload_scan(file: UploadFile = File(...)) -> PredictionResponse:
    if not file.filename:
        raise HTTPException(status_code=400, detail="Uploaded file has no filename.")

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        image_arr = load_rgb_image(content)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}") from exc

    organ = detect_organ_from_filename(file.filename)

    # Brain specialist from Keras image model.
    brain_prob = None
    if registry.models.brain_model is not None:
        try:
            brain_input = np.expand_dims(image_arr, axis=0)
            raw = registry.models.brain_model.predict(brain_input, verbose=0)
            arr = np.asarray(raw).reshape(-1)
            brain_prob = float(arr[0])
            brain_prob = min(max(brain_prob, 0.0), 1.0)
        except Exception:
            brain_prob = None

    tabular_features = image_statistics_features(image_arr)

    liver_features = tabular_features
    if registry.models.liver_scaler is not None:
        try:
            liver_features = registry.models.liver_scaler.transform(tabular_features)
        except Exception:
            liver_features = tabular_features

    liver_rf_prob = registry.predict_binary_probability(registry.models.liver_rf_model, liver_features)
    liver_xgb_prob = registry.predict_binary_probability(registry.models.liver_xgb_model, liver_features)
    liver_lgbm_prob = registry.predict_binary_probability(registry.models.liver_lgbm_model, liver_features)
    health_model_prob = registry.predict_binary_probability(registry.models.health_prediction_model, tabular_features)
    kidney_model_prob = registry.predict_binary_probability(registry.models.kidney_model, tabular_features)

    raw_scores = {
        "brain_tumor_prob": brain_prob,
        "liver_rf_prob": liver_rf_prob,
        "liver_xgb_prob": liver_xgb_prob,
        "liver_lgbm_prob": liver_lgbm_prob,
        "health_model_prob": health_model_prob,
        "kidney_model_prob": kidney_model_prob,
    }
    specialist_scores = normalize_scores(raw_scores)
    final_label, final_prob, model_used = run_meta_model(registry.models.meta_model, specialist_scores)

    return PredictionResponse(
        file_uploaded=file.filename,
        detected_organ=organ,
        model_used=model_used if organ != "Brain" else "Brain Tumor Model",
        prediction=final_label,
        confidence=round(final_prob * 100.0, 2),
        specialist_scores={k: round(v * 100.0, 2) for k, v in specialist_scores.items()},
    )
