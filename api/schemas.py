from datetime import date

from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    date: date
    farm_id: int = Field(gt=0)
    crop: str
    rainfall_mm: float = Field(ge=0)
    temperature_c: float
    soil_moisture: float = Field(ge=0, le=1)
    ndvi: float = Field(ge=0, le=1)
    fertilizer_kg_ha: float = Field(ge=0)
    yield_lag_1: float = Field(default=4.0, ge=0)
    ndvi_rolling_3: float | None = Field(default=None, ge=0, le=1)


class PredictionResponse(BaseModel):
    predicted_yield: float
    model_version: str
