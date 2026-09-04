from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def _fallback(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], predictions: dict | None = None) -> dict:
    rows = []
    for column in columns:
        ref_mean, cur_mean = reference[column].mean(), current[column].mean()
        rows.append({"feature": column, "reference_mean": float(ref_mean), "current_mean": float(cur_mean), "relative_shift": float(abs(cur_mean - ref_mean) / (abs(ref_mean) + 1e-9))})
    output = {"backend": "fallback", "drift_detected": any(row["relative_shift"] > 0.2 for row in rows), "features": rows}
    if predictions:
        output["prediction"] = _prediction_metrics(predictions["reference"], predictions["current"])
        output["prediction_drift_detected"] = output["prediction"]["relative_shift"] > 0.2
    return output


def _prediction_metrics(reference: pd.Series, current: pd.Series) -> dict:
    return {"reference_mean": float(reference.mean()), "current_mean": float(current.mean()), "relative_shift": float(abs(current.mean() - reference.mean()) / (abs(reference.mean()) + 1e-9)), "reference_std": float(reference.std()), "current_std": float(current.std())}


def drift_report(reference: pd.DataFrame, current: pd.DataFrame, columns: list[str], path: str | Path, predictions: dict | None = None) -> dict:
    feature_summary = _fallback(reference, current, columns)
    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
        report = Report([DataDriftPreset()])
        result = report.run(current_data=current[columns], reference_data=reference[columns])
        result.save_html(str(path))
        output = {"backend": "evidently", "drift_detected": feature_summary["drift_detected"], "features": feature_summary["features"], "threshold": 0.2}
        if predictions:
            output["prediction"] = _prediction_metrics(predictions["reference"], predictions["current"])
            output["prediction_drift_detected"] = output["prediction"]["relative_shift"] > 0.2
            if "reference_target" in predictions and "current_target" in predictions:
                output["error"] = {"reference_rmse": float(mean_squared_error(predictions["reference_target"], predictions["reference"] ) ** 0.5), "current_rmse": float(mean_squared_error(predictions["current_target"], predictions["current"]) ** 0.5), "reference_mae": float(mean_absolute_error(predictions["reference_target"], predictions["reference"])), "current_mae": float(mean_absolute_error(predictions["current_target"], predictions["current"]))}
        return output
    except (ImportError, Exception):
        output = _fallback(reference, current, columns, predictions)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).with_suffix(".json").write_text(__import__("json").dumps(output, indent=2), encoding="utf-8")
        return output
