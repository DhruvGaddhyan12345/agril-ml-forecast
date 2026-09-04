from __future__ import annotations

import pandas as pd

REQUIRED_COLUMNS = {"date", "farm_id", "crop", "rainfall_mm", "temperature_c", "soil_moisture", "ndvi", "fertilizer_kg_ha", "yield_tons_per_hectare"}


def validate_data(frame: pd.DataFrame) -> dict:
    missing = sorted(REQUIRED_COLUMNS - set(frame.columns))
    errors = [f"missing columns: {missing}"] if missing else []
    if frame.empty:
        errors.append("dataset is empty")
    if frame.isna().any().any():
        errors.append("dataset contains missing values")
    if frame.select_dtypes(include="number").isin([float("inf"), float("-inf")]).any().any():
        errors.append("dataset contains infinite numeric values")
    if "date" in frame and not pd.to_datetime(frame["date"], errors="coerce").notna().all():
        errors.append("date contains invalid values")
    if {"date", "farm_id", "crop"}.issubset(frame.columns) and frame.duplicated(subset=["date", "farm_id", "crop"]).any():
        errors.append("duplicate date/farm/crop keys")
    if "yield_tons_per_hectare" in frame and frame["yield_tons_per_hectare"].le(0).any():
        errors.append("target must be positive")
    if "farm_id" in frame and (frame["farm_id"] <= 0).any():
        errors.append("farm_id must be positive")
    if "crop" in frame and not frame["crop"].isin(["maize", "wheat", "rice"]).all():
        errors.append("crop contains unsupported values")
    for column in ["rainfall_mm", "temperature_c", "fertilizer_kg_ha"]:
        if column in frame and (frame[column] < 0).any():
            errors.append(f"{column} must not be negative")
    for column in ["soil_moisture", "ndvi"]:
        if column in frame and not frame[column].between(0, 1).all():
            errors.append(f"{column} must be between 0 and 1")
    return {"valid": not errors, "rows": len(frame), "errors": errors}
