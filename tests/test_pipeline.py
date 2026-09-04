import pandas as pd
import pytest

from agriml.data import generate_synthetic_data
from agriml.features import FEATURE_COLUMNS, make_features, temporal_split
from agriml.models import benchmark
from agriml.validation import validate_data


def test_generation_and_validation():
    frame = generate_synthetic_data(200, 7)
    assert validate_data(frame)["valid"]
    assert len(make_features(frame)) > 0


def test_temporal_split_has_no_date_overlap():
    train, test = temporal_split(make_features(generate_synthetic_data(500)))
    assert train.date.max() < test.date.min()


def test_validation_rejects_bad_target():
    frame = generate_synthetic_data(10)
    frame.loc[0, "yield_tons_per_hectare"] = -1
    assert not validate_data(frame)["valid"]


def test_benchmark_has_six_reproducible_models():
    frame = make_features(generate_synthetic_data(500, 11))
    train, test = temporal_split(frame)
    metrics, _, _ = benchmark(train, test, 11)
    assert {"linear", "ridge", "random_forest", "extra_trees", "hist_gradient_boosting", "mlp"}.issubset(metrics["model"])
    assert {"cv_rmse_mean", "cv_rmse_std"}.issubset(metrics.columns)


def test_validation_rejects_missing_and_invalid_values():
    frame = generate_synthetic_data(20)
    frame.loc[0, "rainfall_mm"] = -1
    frame.loc[1, "crop"] = "unknown"
    frame.loc[2, "ndvi"] = None
    result = validate_data(frame)
    assert not result["valid"]
    assert len(result["errors"]) >= 3
