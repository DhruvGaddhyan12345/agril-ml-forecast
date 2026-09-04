import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from agriml.features import make_features, temporal_split
from agriml.models import benchmark

root = Path(__file__).resolve().parents[1]
frame = make_features(pd.read_csv(root / "data/raw/observations.csv", parse_dates=["date"]))
train, test = temporal_split(frame)
metrics, _, _ = benchmark(train, test)
metrics.to_csv(root / "reports/benchmark.csv", index=False)
print(metrics.to_string(index=False))