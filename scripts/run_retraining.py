from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from agriml.features import make_features, temporal_split
from agriml.models import benchmark
from agriml.retraining import controlled_promotion

root = Path(__file__).resolve().parents[1]
frame = make_features(pd.read_csv(root / "data/raw/observations.csv", parse_dates=["date"]))
development, holdout = temporal_split(frame, 0.2)
train, validation = temporal_split(development, 0.2)
current_metrics, current, _ = benchmark(train, holdout)
candidate_metrics, candidate, _ = benchmark(development, holdout)
current_rmse = float(current_metrics.iloc[0]["rmse"])
candidate_result = candidate_metrics.iloc[0].to_dict()
print(controlled_promotion(candidate["model"], candidate_result, root / "models", max_rmse=1.2, baseline_rmse=current_rmse, baseline_model=current["name"], dataset_version="synthetic-seed-42"))