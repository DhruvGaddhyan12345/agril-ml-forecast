from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd

from .features import FEATURE_COLUMNS


class ModelService:
    def __init__(self, model_path: str | Path, log_path: str | Path = "reports/inference.jsonl"):
        self.model_path = Path(model_path)
        self.model = joblib.load(model_path)
        self.log_path = Path(log_path)

    @property
    def model_version(self) -> str:
        return self.model_path.stem

    def predict(self, record: dict) -> float:
        frame = pd.DataFrame([{**record, "yield_lag_1": record.get("yield_lag_1", 4.0), "ndvi_rolling_3": record.get("ndvi_rolling_3", record["ndvi"])}])
        value = float(self.model.predict(frame[FEATURE_COLUMNS])[0])
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.log_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "region": record.get("farm_id"), "crop": record.get("crop"), "features": record, "prediction": value, "model_version": self.model_version}, default=str) + "\n")
        return value
