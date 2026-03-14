from pydantic import BaseModel


class PredictionResponse(BaseModel):
    file_uploaded: str
    detected_organ: str
    model_used: str
    prediction: str
    confidence: float
    specialist_scores: dict[str, float]


class HealthResponse(BaseModel):
    status: str
