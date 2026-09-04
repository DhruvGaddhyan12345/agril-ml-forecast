import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from agriml.features import FEATURE_COLUMNS, make_features, temporal_split
from agriml.monitoring import drift_report
from agriml.inference import ModelService

root = Path(__file__).resolve().parents[1]
frame = make_features(pd.read_csv(root / "data/raw/observations.csv", parse_dates=["date"]))
reference, normal = temporal_split(frame)
shifted = normal.copy()
shifted["rainfall_mm"] += 20
model = ModelService(root / "models/production.joblib")
reference_predictions = model.model.predict(reference[FEATURE_COLUMNS])
normal_predictions = model.model.predict(normal[FEATURE_COLUMNS])
shifted_predictions = model.model.predict(shifted[FEATURE_COLUMNS])
result = {"normal": drift_report(reference, normal, FEATURE_COLUMNS, root / "reports/drift_normal.html", {"reference": pd.Series(reference_predictions), "current": pd.Series(normal_predictions), "reference_target": reference["yield_tons_per_hectare"], "current_target": normal["yield_tons_per_hectare"]}), "shifted": drift_report(reference, shifted, FEATURE_COLUMNS, root / "reports/drift_shifted.html", {"reference": pd.Series(reference_predictions), "current": pd.Series(shifted_predictions), "reference_target": reference["yield_tons_per_hectare"], "current_target": shifted["yield_tons_per_hectare"]})}
(root / "reports/drift_experiment.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
print(json.dumps(result, indent=2))