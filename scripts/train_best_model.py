import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd
from agriml.features import make_features, temporal_split
from agriml.models import benchmark

root = Path(__file__).resolve().parents[1]
frame = make_features(pd.read_csv(root / "data/raw/observations.csv", parse_dates=["date"]))
train, test = temporal_split(frame)
metrics, best, _ = benchmark(train, test)
root.joinpath("models").mkdir(exist_ok=True)
joblib.dump(best["model"], root / "models/candidate.joblib")
print(metrics.iloc[0].to_dict())