import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agriml.data import write_synthetic_data

root = Path(__file__).resolve().parents[1]
write_synthetic_data(root / "data/raw/observations.csv")
print("Wrote data/raw/observations.csv")