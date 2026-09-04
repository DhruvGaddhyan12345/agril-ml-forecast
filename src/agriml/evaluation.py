from __future__ import annotations

from pathlib import Path

import pandas as pd


def error_report(test: pd.DataFrame, predictions, path: str | Path) -> pd.DataFrame:
    report = test[["date", "farm_id", "crop", "yield_tons_per_hectare"]].copy()
    report["prediction"] = predictions
    report["error"] = report["prediction"] - report["yield_tons_per_hectare"]
    report["absolute_error"] = report["error"].abs()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    return report
