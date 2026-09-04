from __future__ import annotations

import pandas as pd

FEATURE_COLUMNS = ["rainfall_mm", "temperature_c", "soil_moisture", "ndvi", "fertilizer_kg_ha", "yield_lag_1", "ndvi_rolling_3"]


def make_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.sort_values(["farm_id", "crop", "date"]).copy()
    groups = result.groupby(["farm_id", "crop"], sort=False)
    result["yield_lag_1"] = groups["yield_tons_per_hectare"].shift(1)
    result["ndvi_rolling_3"] = groups["ndvi"].transform(lambda values: values.shift(1).rolling(3, min_periods=1).mean())
    return result.dropna(subset=FEATURE_COLUMNS).sort_values("date").reset_index(drop=True)


def temporal_split(frame: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = frame.sort_values("date").reset_index(drop=True)
    cutoff = ordered["date"].quantile(1 - test_fraction)
    train, test = ordered[ordered.date < cutoff], ordered[ordered.date >= cutoff]
    if train.empty or test.empty or train.date.max() >= test.date.min():
        raise ValueError("temporal split must have non-overlapping train and test dates")
    return train, test
