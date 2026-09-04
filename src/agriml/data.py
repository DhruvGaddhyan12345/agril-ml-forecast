from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_data(n_rows: int = 2400, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=730, freq="D")
    keys = rng.choice(len(dates) * 12 * 3, n_rows, replace=False)
    date = dates[keys // 36]
    farms = keys // 3 % 12 + 1
    crops = np.array(["maize", "wheat", "rice"])[keys % 3]
    temperature = 21 + 8 * np.sin(2 * np.pi * pd.Series(date).dt.dayofyear.to_numpy() / 365) + rng.normal(0, 2, n_rows)
    rainfall = np.maximum(0, rng.gamma(2.0, 12.0, n_rows))
    soil_moisture = np.clip(0.28 + rainfall / 180 + rng.normal(0, 0.04, n_rows), 0.08, 0.75)
    ndvi = np.clip(0.55 + 0.12 * soil_moisture + rng.normal(0, 0.05, n_rows), 0.1, 0.95)
    fertilizer = np.clip(rng.normal(95, 25, n_rows), 10, 180)
    crop_effect = pd.Series(crops).map({"maize": 0.5, "wheat": 0.2, "rice": 0.8}).to_numpy()
    yield_value = 2.2 + crop_effect + 2.8 * ndvi + 0.012 * fertilizer - 0.018 * (temperature - 23) ** 2 + rng.normal(0, 0.22, n_rows)
    return pd.DataFrame({"date": pd.to_datetime(date), "farm_id": farms, "crop": crops, "rainfall_mm": rainfall, "temperature_c": temperature, "soil_moisture": soil_moisture, "ndvi": ndvi, "fertilizer_kg_ha": fertilizer, "yield_tons_per_hectare": np.maximum(0.2, yield_value)}).sort_values("date").reset_index(drop=True)


def write_synthetic_data(path: str | Path, n_rows: int = 2400, seed: int = 42) -> pd.DataFrame:
    frame = generate_synthetic_data(n_rows, seed)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return frame
