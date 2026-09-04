import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agriml.data import generate_synthetic_data
from agriml.features import make_features, temporal_split
from agriml.models import benchmark
from agriml.retraining import controlled_promotion

root = Path(__file__).resolve().parents[1]
frame = make_features(generate_synthetic_data(2400, 43))
train, test = temporal_split(frame)
metrics, best, _ = benchmark(train, test, 43)
result = controlled_promotion(best["model"], metrics.iloc[0].to_dict(), root / "models", max_rmse=1.2)
print(json.dumps(result, indent=2))
