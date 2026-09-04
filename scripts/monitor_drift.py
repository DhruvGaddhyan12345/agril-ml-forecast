from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from agriml.features import FEATURE_COLUMNS, make_features, temporal_split
from agriml.monitoring import drift_report

root = Path(__file__).resolve().parents[1]
frame = make_features(pd.read_csv(root / "data/raw/observations.csv", parse_dates=["date"]))
reference, current = temporal_split(frame)
print(drift_report(reference, current, FEATURE_COLUMNS, root / "reports/drift.html"))
