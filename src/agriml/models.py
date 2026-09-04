from __future__ import annotations

import importlib.util
from time import perf_counter

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit

from .features import FEATURE_COLUMNS


class TorchMLPRegressor:
    """Small CPU PyTorch regressor with an sklearn-compatible interface."""

    def __init__(self, seed: int = 42, epochs: int = 120):
        self.seed = seed
        self.epochs = epochs

    def fit(self, features: pd.DataFrame, target: pd.Series):
        import torch
        from torch import nn

        torch.manual_seed(self.seed)
        self.network = nn.Sequential(nn.Linear(features.shape[1], 32), nn.ReLU(), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))
        optimizer = torch.optim.Adam(self.network.parameters(), lr=0.01)
        loss_fn = nn.MSELoss()
        inputs = torch.tensor(features.to_numpy(dtype=np.float32))
        labels = torch.tensor(target.to_numpy(dtype=np.float32)).reshape(-1, 1)
        self.network.train()
        for _ in range(self.epochs):
            optimizer.zero_grad()
            loss_fn(self.network(inputs), labels).backward()
            optimizer.step()
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        import torch

        self.network.eval()
        with torch.no_grad():
            return self.network(torch.tensor(features.to_numpy(dtype=np.float32))).numpy().reshape(-1)


def benchmark(train: pd.DataFrame, test: pd.DataFrame, seed: int = 42) -> tuple[pd.DataFrame, dict, dict]:
    candidates = {
        "linear": LinearRegression(),
        "ridge": Ridge(alpha=1.0),
        "random_forest": RandomForestRegressor(n_estimators=80, random_state=seed, n_jobs=-1, min_samples_leaf=2),
        "extra_trees": ExtraTreesRegressor(n_estimators=80, random_state=seed, n_jobs=-1, min_samples_leaf=2),
        "hist_gradient_boosting": HistGradientBoostingRegressor(max_iter=120, random_state=seed),
        "mlp": MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, early_stopping=True, random_state=seed),
    }
    optional = {"xgboost": ("xgboost", "XGBRegressor"), "lightgbm": ("lightgbm", "LGBMRegressor")}
    for name, (module, cls) in optional.items():
        if importlib.util.find_spec(module):
            model_class = getattr(__import__(module, fromlist=[cls]), cls)
            candidates[name] = model_class(n_estimators=80, random_state=seed, verbosity=0)
    if importlib.util.find_spec("torch"):
        candidates["pytorch_mlp"] = TorchMLPRegressor(seed=seed)
    metrics, fitted = [], {}
    for name, model in candidates.items():
        started = perf_counter()
        model.fit(train[FEATURE_COLUMNS], train["yield_tons_per_hectare"])
        predictions = model.predict(test[FEATURE_COLUMNS])
        cv_scores = []
        splitter = TimeSeriesSplit(n_splits=3)
        for cv_train, cv_valid in splitter.split(train):
            fold_model = clone(model) if name != "pytorch_mlp" else TorchMLPRegressor(seed=seed)
            fold_model.fit(train.iloc[cv_train][FEATURE_COLUMNS], train.iloc[cv_train]["yield_tons_per_hectare"])
            cv_prediction = fold_model.predict(train.iloc[cv_valid][FEATURE_COLUMNS])
            cv_scores.append(float(np.sqrt(mean_squared_error(train.iloc[cv_valid]["yield_tons_per_hectare"], cv_prediction))))
        metrics.append({"model": name, "rmse": float(np.sqrt(mean_squared_error(test.yield_tons_per_hectare, predictions))), "mae": float(mean_absolute_error(test.yield_tons_per_hectare, predictions)), "r2": float(r2_score(test.yield_tons_per_hectare, predictions)), "cv_rmse_mean": float(np.mean(cv_scores)), "cv_rmse_std": float(np.std(cv_scores)), "fit_seconds": perf_counter() - started})
        fitted[name] = (model, predictions)
    report = pd.DataFrame(metrics).sort_values("rmse").reset_index(drop=True)
    best_name = str(report.iloc[0].model)
    return report, {"name": best_name, "model": fitted[best_name][0], "predictions": fitted[best_name][1]}, fitted
