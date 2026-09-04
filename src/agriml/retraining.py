from __future__ import annotations

import json
from pathlib import Path

import joblib
from datetime import datetime, timezone


def controlled_promotion(model, metrics: dict, model_dir: str | Path, max_rmse: float = 1.2, min_improvement: float = 0.0, baseline_rmse: float | None = None, baseline_model: str | None = None, dataset_version: str | None = None) -> dict:
    approved = metrics["rmse"] <= max_rmse and (baseline_rmse is None or baseline_rmse - metrics["rmse"] >= min_improvement)
    directory = Path(model_dir)
    directory.mkdir(parents=True, exist_ok=True)
    candidate = directory / "candidate.joblib"
    previous_production = directory / "production.joblib"
    joblib.dump(model, candidate)
    if approved:
        joblib.dump(model, directory / "production.joblib")
    improvement = None if baseline_rmse is None else float(baseline_rmse - metrics["rmse"])
    result = {"approved": approved, "decision": "PROMOTE" if approved else "KEEP_CURRENT", "candidate_metrics": metrics, "baseline_rmse": baseline_rmse, "improvement_rmse": improvement, "baseline_model": baseline_model, "dataset_version": dataset_version, "decision_timestamp": datetime.now(timezone.utc).isoformat(), "threshold": max_rmse, "candidate": str(candidate), "previous_production_exists": previous_production.exists(), "production": str(directory / "production.joblib") if approved else None}
    (directory / "promotion.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result
