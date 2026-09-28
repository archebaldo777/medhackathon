from __future__ import annotations

import importlib.util
from pathlib import Path

import joblib
import numpy as np
import pytest

from src.analyzer import is_borderline
from src.features import feature_sizes
from src.model import DEFAULT_MODEL_PATH, QualityModel, resolve_threshold


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "dxa_train", PROJECT_ROOT / "scripts" / "train.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_interior_threshold_is_kept():
    assert resolve_threshold({"threshold": 0.23, "oof_threshold": 0.31}) == (0.23, "rate_matched")


def test_unreachable_threshold_falls_back_to_validated_operating_point():
    """A transferred threshold of 1.0 only fires on a unanimous ensemble vote."""
    assert resolve_threshold({"threshold": 1.0, "oof_threshold": 0.29}) == (0.29, "out_of_fold")
    assert resolve_threshold({"threshold": 0.0, "oof_threshold": 0.29}) == (0.29, "out_of_fold")


def test_threshold_falls_back_to_neutral_without_a_validated_value():
    assert resolve_threshold({"threshold": 1.0}) == (0.5, "neutral")
    assert resolve_threshold({"threshold": 0.0, "oof_threshold": 0.0}) == (0.5, "neutral")


@pytest.mark.skipif(not DEFAULT_MODEL_PATH.exists(), reason="модель не собрана")
def test_shipped_model_has_no_degenerate_operating_point():
    model = QualityModel()
    assert set(model.thresholds) == set(model.bundle["classifiers"])
    for key, threshold in model.thresholds.items():
        assert 0.0 < threshold < 1.0, f"{key} не может сработать: порог {threshold}"


def test_borderline_band_follows_the_decision_threshold():
    # Just under the operating point of a head calibrated far below 0.5.
    assert is_borderline(0.080, 0.0869)
    # Comfortably negative for that head, even though it is near 0.5.
    assert not is_borderline(0.5, 0.0869)
    # At or above the threshold the result is already positive.
    assert not is_borderline(0.0869, 0.0869)
    assert not is_borderline(0.2, 0.0869)
    # The band scales with a high threshold too.
    assert is_borderline(0.27, 0.29)
    assert not is_borderline(0.1, 0.29)


def test_rate_matching_rejects_saturated_in_sample_probabilities(tmp_path):
    train = load_train_module()
    # A memorising estimator scores its own training set as 0 or 1 only.
    saturated = np.asarray([0.0] * 55 + [1.0] * 45)
    threshold, source = train.rate_matched_threshold(saturated, 0.4767, fallback=0.29)
    assert source == "out_of_fold"
    assert threshold == 0.29

    graded = np.linspace(0.0, 1.0, 101)
    threshold, source = train.rate_matched_threshold(graded, 0.5, fallback=0.29)
    assert source == "rate_matched"
    assert 0.0 < threshold < 1.0


def _linear_head(size: int, threshold: float, oof_threshold: float) -> dict:
    return {
        "estimator_type": "linear",
        "threshold": threshold,
        "oof_threshold": oof_threshold,
        "pooling": "none",
        "feature_size": size,
        "coef": np.zeros(size, dtype=np.float32),
        "intercept": 0.0,
        "feature_mean": np.zeros(size, dtype=np.float32),
        "feature_scale": np.ones(size, dtype=np.float32),
    }


def test_model_applies_resolved_threshold_to_predictions(tmp_path):
    """The resolved operating point, not the stored one, decides the class."""
    sizes = feature_sizes()
    bundle = {
        "format_version": 4,
        "model_version": "test",
        "feature_sizes": sizes,
        "classifiers": {
            # Unreachable stored value on one head; the validated one must be used.
            key: _linear_head(size, 1.0 if key == "spine_quality" else 0.3, 0.2)
            for key, size in sizes.items()
        },
    }
    path = tmp_path / "bundle.joblib"
    joblib.dump(bundle, path)
    model = QualityModel(path)
    assert model.thresholds["spine_quality"] == 0.2
    assert model.threshold_sources["spine_quality"] == "out_of_fold"
    assert model.thresholds["hip_quality"] == 0.3
    # A zero-weight linear head outputs 0.5, which clears 0.2 but never 1.0.
    zeros = np.zeros(sizes["spine_quality"], dtype=np.float32)
    assert model._probability("spine_quality", zeros) == 0.5


def test_model_rejects_a_bundle_built_for_different_features(tmp_path):
    """A stale model must fail loudly instead of scoring the wrong vector."""
    sizes = feature_sizes()
    bundle = {
        "format_version": 4,
        "model_version": "test",
        "feature_sizes": sizes,
        "classifiers": {key: _linear_head(size + 1, 0.3, 0.2) for key, size in sizes.items()},
    }
    path = tmp_path / "stale.joblib"
    joblib.dump(bundle, path)
    with pytest.raises(ValueError, match="не соответствует коду признаков"):
        QualityModel(path)
