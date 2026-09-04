import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from agriml.validation import validate_data

root = Path(__file__).resolve().parents[1]
result = validate_data(pd.read_csv(root / "data/raw/observations.csv"))
print(result)
if not result["valid"]:
    raise SystemExit(1)