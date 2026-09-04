from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import joblib
import pandas as pd
import yaml

from agriml.data import write_synthetic_data
from agriml.evaluation import error_report
from agriml.features import FEATURE_COLUMNS, make_features, temporal_split
from agriml.models import benchmark
from agriml.monitoring import drift_report
from agriml.retraining import controlled_promotion
from agriml.tracking import log_run
from agriml.validation import validate_data


def main():
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "configs/default.yaml").read_text(encoding="utf-8"))
    raw_path = root / config["data_dir"] / "raw" / "observations.csv"
    report_dir, model_dir = root / config["report_dir"], root / config["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    raw = write_synthetic_data(raw_path, config["n_rows"], config["seed"])
    validation = validate_data(raw)
    (report_dir).mkdir(exist_ok=True)
    (report_dir / "validation.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    if not validation["valid"]:
        raise ValueError(validation["errors"])
    featured = make_features(raw)
    train, test = temporal_split(featured, config["test_fraction"])
    metrics, best, _ = benchmark(train, test, config["seed"])
    metrics.to_csv(report_dir / "benchmark.csv", index=False)
    best_metrics = metrics.iloc[0].to_dict()
    joblib.dump(best["model"], model_dir / "candidate.joblib")
    errors = error_report(test, best["predictions"], report_dir / "errors.csv")
    promotion = controlled_promotion(best["model"], best_metrics, model_dir, **config["promotion"])
    tracking = log_run(best_metrics, {"model": best["name"], "train_rows": len(train)})
    drift = drift_report(train, test, FEATURE_COLUMNS, report_dir / "drift.html", {"reference": pd.Series(best["model"].predict(train[FEATURE_COLUMNS])), "current": pd.Series(best["predictions"]), "reference_target": train["yield_tons_per_hectare"], "current_target": test["yield_tons_per_hectare"]})
    summary = {"rows": len(raw), "train_rows": len(train), "test_rows": len(test), "best_model": best["name"], "metrics": best_metrics, "mean_absolute_error": float(errors.absolute_error.mean()), "promotion": promotion, "tracking": tracking, "drift": drift}
    (report_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
