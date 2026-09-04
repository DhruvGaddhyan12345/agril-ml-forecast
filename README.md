# AgriML

AgriML is an end-to-end crop-yield forecasting and MLOps platform. It generates deterministic synthetic agricultural data, validates it, performs a leakage-safe temporal split and feature engineering, benchmarks six built-in scikit-learn regressors plus XGBoost, LightGBM, and a real PyTorch MLP, tracks runs with MLflow, monitors feature and prediction drift with Evidently, retrains candidates, and promotes a model only when a measured quality gate passes.

## Quick start

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev,full]"
python scripts/run_full_pipeline.py
pytest
```

The pipeline writes generated artifacts to `data/`, `models/`, and `reports/`. Metrics and drift values are computed during execution and are never embedded as fixtures. Start the API with `uvicorn api.main:app --reload` from the repository root and query `GET /health` or `POST /predict`.

The base install provides the six deterministic scikit-learn models. The `full` profile adds XGBoost, LightGBM, PyTorch, MLflow, and Evidently so the complete roadmap benchmark is reproducible. Python 3.11 is the validated environment for the full stack.

## Verified benchmark

The full environment benchmarks nine models and writes measured metrics to `reports/benchmark.csv`. The best model is selected by RMSE; no benchmark values are hard-coded.

## Monitoring and lifecycle

The drift experiment reports feature drift, prediction-distribution drift, and normal-versus-shifted RMSE/MAE. Candidate retraining evaluates current and candidate models on the same final holdout. Promotion writes an auditable `models/promotion.json` decision and never promotes solely because drift was detected.

## Docker

```powershell
docker compose up --build
```

The entrypoint generates a production model automatically if the mounted `models/` directory is empty. Docker Desktop must be running.

## Project layout

`src/agriml` contains the domain code, `scripts` contains executable workflows, `configs` contains YAML configuration, `tests` covers the important contracts, and `api` exposes inference.
