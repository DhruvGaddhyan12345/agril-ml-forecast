import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from agriml.inference import ModelService

root = Path(__file__).resolve().parents[1]
service = ModelService(root / "models/production.joblib", root / "reports/inference.jsonl")
frame = pd.read_csv(root / "data/raw/observations.csv").head(25)
for record in frame.to_dict(orient="records"):
    service.predict(record)
print(json.dumps({"records": len(frame), "model_version": service.model_version}))