# Medical AI Diagnosis Platform

This repository contains a full-stack medical AI diagnosis system:

- Frontend: single static HTML file (`frontend/index.html`), no build step required.
- Django (`:8000`): request proxy + prediction history database.
- FastAPI (`:8001`): model loading and ML inference.

## File Structure

- `fastapi_server/main.py`
- `fastapi_server/requirements.txt`
- `django_backend/core/settings.py`
- `django_backend/core/urls.py`
- `django_backend/predictions/models.py`
- `django_backend/predictions/serializers.py`
- `django_backend/predictions/views.py`
- `django_backend/predictions/urls.py`
- `django_backend/requirements.txt`
- `frontend/index.html`

## Models Used

Loaded from `model/` at FastAPI startup:

- `brain_tumor_model.h5`, `brain_tumor.h5`
- `health_prediction_model.pkl`
- `kidney_model.pkl`
- `liver_label_encoder.joblib`
- `liver_scaler.joblib`
- `liver_lgbm_model.joblib`
- `liver_rf_model.joblib`
- `liver_xgb_model.joblib`
- `meta_model.joblib`

## Why Confidence Was Showing 0%

The 0% bug happens when code reads `model.predict_proba(X)[0]` as if it were a scalar.

For binary classifiers, `predict_proba(X)[0]` is usually an array like `[p0, p1]`.
If this array is cast or handled incorrectly, confidence can collapse to `0`.

Correct extraction is:

- sklearn binary models: `float(model.predict_proba(X)[0, 1])`
- Keras sigmoid output `(N, 1)`: `float(raw[0, 0])`
- Keras softmax output `(N, classes)`: `float(raw[0, 1])`

This logic is implemented in `get_proba()` inside `fastapi_server/main.py`.

## API Endpoints

### FastAPI (`http://127.0.0.1:8001`)

- `GET /health`
- `POST /predict/brain` (multipart image)
- `POST /predict/kidney` (JSON list of 24 values OR `{ "features": [...] }`)
- `POST /predict/liver` (JSON 10 fields)
- `POST /predict/health` (JSON list OR `{ "features": [...] }`)

Each response includes:

- `organ`
- `model`
- `prediction`
- `confidence` (0-100)
- `risk_level` (`Low`, `Medium`, `High`)
- per-model breakdown fields where relevant

### Django (`http://127.0.0.1:8000/api`)

- `POST /predict/brain/`
- `POST /predict/kidney/`
- `POST /predict/liver/`
- `POST /predict/health/`
- `GET /history/`
- `GET /stats/`
- `GET /health/`

Django saves every prediction to `PredictionRecord` and enriches response with:

- `record_id`
- `saved_at`

## Startup Commands

### 1) FastAPI

```bash
pip install -r fastapi_server/requirements.txt
python -m uvicorn fastapi_server.main:app --host 127.0.0.1 --port 8001 --reload
```

### 2) Django

```bash
pip install -r django_backend/requirements.txt
cd django_backend
python manage.py migrate
python manage.py runserver 127.0.0.1:8000
```

### 3) Frontend (no build step)

From project root, start a static file server:

```bash
python -m http.server 5173 -d frontend
```

Then open:

- `http://127.0.0.1:5173/index.html`

The frontend script points to:

- `DJANGO_URL = "http://localhost:8000/api"`