import pandas as pd
from pathlib import Path

from agriml.monitoring import drift_report


def test_monitoring_reports_prediction_and_error_drift():
    reference = pd.DataFrame({"feature": [1.0, 1.0, 1.0]})
    current = pd.DataFrame({"feature": [2.0, 2.0, 2.0]})
    result = drift_report(
        reference,
        current,
        ["feature"],
        Path("reports/test_drift.html"),
        {"reference": pd.Series([1.0, 1.0, 1.0]), "current": pd.Series([2.0, 2.0, 2.0]), "reference_target": pd.Series([1.0, 1.0, 1.0]), "current_target": pd.Series([1.0, 1.0, 1.0])},
    )
    assert result["drift_detected"]
    assert result["prediction_drift_detected"]
    assert result["error"]["current_rmse"] > result["error"]["reference_rmse"]