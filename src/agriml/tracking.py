from __future__ import annotations


def log_run(metrics: dict, params: dict, experiment: str = "agriml") -> dict:
    try:
        import mlflow
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=params.get("model", "benchmark")):
            mlflow.log_params(params)
            mlflow.log_metrics({key: value for key, value in metrics.items() if isinstance(value, (int, float))})
        return {"enabled": True, "backend": "mlflow"}
    except (ImportError, Exception) as error:
        return {"enabled": False, "backend": "local", "reason": type(error).__name__}
