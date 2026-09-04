from pathlib import Path

from fastapi import FastAPI, HTTPException

from agriml.inference import ModelService
from .schemas import PredictionRequest, PredictionResponse

app = FastAPI(title="AgriML API", version="0.1.0")
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/production.joblib"
LOG_PATH = ROOT / "reports/inference.jsonl"


@app.get("/health")
def health():
    return {"status": "ok", "model_ready": MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="production model is not available")
    payload = request.model_dump()
    if payload["ndvi_rolling_3"] is None:
        payload["ndvi_rolling_3"] = payload["ndvi"]
    try:
        service = ModelService(MODEL_PATH, LOG_PATH)
        return {"predicted_yield": service.predict(payload), "model_version": service.model_version}
    except (KeyError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
