from __future__ import annotations

import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from .constants import (
    DEFAULT_PIXEL_SPACING_X_MM,
    DEFAULT_PIXEL_SPACING_Y_MM,
    HIP_LABELS,
    LEFT_HIP,
    RIGHT_HIP,
    SPINE,
    SPINE_LABELS,
    VIOLATIONS,
)
from .features import extract_features, feature_sizes


MODEL_FILE_NAME = "dxa_quality_v4.joblib"


def default_model_path() -> Path:
    """Model weights: $DXA_MODEL_PATH, then models/ next to the source tree, then ./models."""
    override = os.environ.get("DXA_MODEL_PATH")
    if override:
        return Path(override)
    candidates = (
        Path(__file__).resolve().parent.parent / "models" / MODEL_FILE_NAME,
        Path.cwd() / "models" / MODEL_FILE_NAME,
    )
    return next((path for path in candidates if path.exists()), candidates[0])


DEFAULT_MODEL_PATH = default_model_path()
NEUTRAL_THRESHOLD = 0.5


# Every head is rescaled so that its own operating point lands on this value.
DECISION_POINT = 0.5


def align_to_threshold(probability: float, threshold: float) -> float:
    """Monotone map of one head's probability that sends its threshold to 0.5.

    The heads are calibrated at very different operating points (0.15 for one,
    0.85 for another), so their raw probabilities cannot be compared or combined.
    After this map every head says "violation" exactly above 0.5, which lets the
    report carry one probability that agrees with the exported class.
    """
    probability = min(max(float(probability), 0.0), 1.0)
    if threshold <= 0.0:
        return 1.0
    if threshold >= 1.0:
        return DECISION_POINT * probability
    if probability < threshold:
        return DECISION_POINT * probability / threshold
    return DECISION_POINT + (1.0 - DECISION_POINT) * (probability - threshold) / (1.0 - threshold)


def resolve_threshold(item: dict[str, Any]) -> tuple[float, str]:
    """Pick a usable operating point for one head.

    ``threshold`` is the training-time value transferred to the refit estimator
    by matching the out-of-fold positive rate. When that estimator memorises its
    training set — deep trees in particular — the in-sample probabilities
    saturate and the transferred value lands on the edge of the range. A stored
    threshold of ``1.0`` then fires only on a unanimous ensemble vote and one of
    ``0.0`` flags every image, so in both cases the head no longer reproduces the
    operating point it was validated at. The out-of-fold threshold saved
    alongside it is the honest fallback, because it was measured on predictions
    for images the fold model had never seen.
    """
    deployment = float(item.get("threshold", NEUTRAL_THRESHOLD))
    if 0.0 < deployment < 1.0:
        return deployment, "rate_matched"
    validated = item.get("oof_threshold")
    if validated is not None and 0.0 < float(validated) < 1.0:
        return float(validated), "out_of_fold"
    return NEUTRAL_THRESHOLD, "neutral"


@dataclass(frozen=True)
class ModelResult:
    region: str
    quality_class: int
    quality_probability: float
    violation_keys: tuple[str, ...]
    probabilities: dict[str, float]
    model_version: str
    decision_threshold: float
    head_thresholds: dict[str, float] | None = None
    aggregate_probability: float | None = None

    @property
    def violation_codes(self) -> list[str]:
        return [VIOLATIONS[key].code for key in self.violation_keys]

    @property
    def violation_labels(self) -> list[str]:
        return [VIOLATIONS[key].title_ru for key in self.violation_keys]

    @property
    def explanations(self) -> list[dict[str, Any]]:
        return [
            {
                "code": VIOLATIONS[key].code,
                "title": VIOLATIONS[key].title_ru,
                "description": VIOLATIONS[key].explanation_ru,
                "probability": round(self.probabilities.get(key, self.quality_probability), 4),
                "threshold": round((self.head_thresholds or {}).get(key, DECISION_POINT), 4),
            }
            for key in self.violation_keys
        ]


class QualityModel:
    def __init__(self, model_path: str | Path = DEFAULT_MODEL_PATH):
        self.path = Path(model_path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"Файл модели не найден: {self.path}. Выполните `make train` "
                "(или scripts/train.py) — обучение занимает около минуты."
            )
        self.bundle: dict[str, Any] = joblib.load(self.path)
        if self.bundle.get("format_version") != 4:
            raise ValueError(
                "Неподдерживаемая версия файла модели: ожидается format_version=4. "
                "Пересоберите модель командой `make train` — начиная с версии 4 каждая "
                "голова использует собственный набор признаков."
            )
        expected = feature_sizes()
        for key, item in self.bundle["classifiers"].items():
            stored = int(item.get("feature_size", -1))
            if stored != expected.get(key):
                raise ValueError(
                    f"Модель не соответствует коду признаков: голова {key} обучена на "
                    f"{stored} признаках, текущий код даёт {expected.get(key)}. "
                    "Выполните `make train`."
                )
        resolved = {
            key: resolve_threshold(item)
            for key, item in self.bundle["classifiers"].items()
        }
        self.thresholds: dict[str, float] = {key: value for key, (value, _) in resolved.items()}
        self.threshold_sources: dict[str, str] = {
            key: source for key, (_, source) in resolved.items()
        }

    @staticmethod
    def _sigmoid(value: float) -> float:
        value = max(-35.0, min(35.0, value))
        return 1.0 / (1.0 + math.exp(-value))

    def _probability(self, key: str, features: np.ndarray) -> float:
        item = self.bundle["classifiers"][key]
        estimator_type = item.get("estimator_type", "linear")
        if estimator_type == "extra_trees":
            return float(item["estimator"].predict_proba(features.reshape(1, -1))[0, 1])
        mean = np.asarray(item.get("feature_mean", self.bundle.get("feature_mean")), dtype=np.float32)
        scale = np.asarray(item.get("feature_scale", self.bundle.get("feature_scale")), dtype=np.float32)
        standardized = (features - mean) / np.where(scale > 1e-6, scale, 1.0)
        if estimator_type != "linear":
            return float(item["estimator"].predict_proba(standardized.reshape(1, -1))[0, 1])
        score = float(np.dot(np.asarray(item["coef"], dtype=np.float32), standardized))
        score += float(item["intercept"])
        return self._sigmoid(score)

    def _probabilities(self, key: str, feature_rows: np.ndarray) -> list[float]:
        """Probabilities for many feature vectors of one head in a single call.

        Tree ensembles are evaluated as one matrix: a call per vector pays the
        estimator's parallel dispatch overhead every time, which dominated the
        runtime of the occlusion heatmap. Per-row values are identical to
        ``_probability``. Linear heads stay per row to keep the arithmetic exact.
        """
        rows = np.atleast_2d(np.asarray(feature_rows, dtype=np.float32))
        item = self.bundle["classifiers"][key]
        estimator_type = item.get("estimator_type", "linear")
        if estimator_type == "extra_trees":
            return [float(value) for value in item["estimator"].predict_proba(rows)[:, 1]]
        return [self._probability(key, row) for row in rows]

    def predict(
        self,
        normalized: np.ndarray,
        region: str,
        spacing_y_mm: float = DEFAULT_PIXEL_SPACING_Y_MM,
        spacing_x_mm: float = DEFAULT_PIXEL_SPACING_X_MM,
    ) -> ModelResult:
        return self.predict_group([normalized], region, spacing_y_mm, spacing_x_mm)[0]

    def predict_group(
        self,
        normalized_images: list[np.ndarray],
        region: str,
        spacing_y_mm: float = DEFAULT_PIXEL_SPACING_Y_MM,
        spacing_x_mm: float = DEFAULT_PIXEL_SPACING_X_MM,
    ) -> list[ModelResult]:
        """Predict a study/side series while preserving per-image heads.

        Heads trained with series pooling share a probability across the group;
        image-level heads retain a separate probability per frame. A single image
        is a valid one-frame group. Every head builds its own descriptor, so the
        features are computed per target rather than once per image.
        """
        if not normalized_images:
            return []
        if region == SPINE:
            quality_key = "spine_quality"
            label_keys = SPINE_LABELS
        elif region in (RIGHT_HIP, LEFT_HIP):
            quality_key = "hip_quality"
            label_keys = HIP_LABELS
        else:
            raise ValueError(f"Неизвестная анатомическая область: {region}")

        probability_rows: list[dict[str, float]] = [dict() for _ in normalized_images]
        for key in (quality_key, *label_keys):
            feature_rows = np.stack(
                [
                    extract_features(normalized, region, key, spacing_y_mm, spacing_x_mm)
                    for normalized in normalized_images
                ]
            )
            pooling = self.bundle["classifiers"][key].get("pooling", "none")
            if pooling == "none":
                values = self._probabilities(key, feature_rows)
            else:
                pooled = getattr(feature_rows, pooling)(axis=0)
                values = [self._probability(key, pooled)] * len(normalized_images)
            for row, value in zip(probability_rows, values):
                row[key] = value

        heads = (quality_key, *label_keys)
        head_thresholds = {key: self.thresholds[key] for key in heads}
        results = []
        for probabilities in probability_rows:
            violations = tuple(
                key for key in label_keys if probabilities[key] >= self.thresholds[key]
            )
            # The reported probability is the strongest head after aligning every
            # head's operating point to 0.5, so it exceeds 0.5 exactly when some
            # head fires. quality_class and quality_prob can no longer disagree,
            # and the organiser's ROC-AUC is computed on a score that ranks by the
            # same evidence the class uses (OOF: spine 0.760 -> 0.840).
            combined = max(
                align_to_threshold(probabilities[key], self.thresholds[key]) for key in heads
            )
            quality = int(combined >= DECISION_POINT)
            results.append(
                ModelResult(
                    region=region,
                    quality_class=quality,
                    quality_probability=combined,
                    violation_keys=violations,
                    probabilities=probabilities,
                    model_version=str(self.bundle.get("model_version", "unknown")),
                    decision_threshold=DECISION_POINT,
                    head_thresholds=head_thresholds,
                    aggregate_probability=probabilities[quality_key],
                )
            )
        return results

    def heatmap(
        self,
        normalized: np.ndarray,
        region: str,
        result: ModelResult,
        spacing_y_mm: float = DEFAULT_PIXEL_SPACING_Y_MM,
        spacing_x_mm: float = DEFAULT_PIXEL_SPACING_X_MM,
    ) -> np.ndarray:
        """Where the decision came from, measured by occluding the image.

        Each head now reads its own geometric descriptor rather than a shared
        appearance grid, so linear coefficients no longer map onto image pixels.
        Occlusion answers the same question for every estimator family: how much
        does the head's probability move when this patch is removed.
        """
        if result.violation_keys:
            key = result.violation_keys[0]
        else:
            key = "spine_quality" if region == SPINE else "hip_quality"
        return self._occlusion_heatmap(normalized, region, key, spacing_y_mm, spacing_x_mm)

    def _occlusion_heatmap(
        self,
        normalized: np.ndarray,
        region: str,
        key: str,
        spacing_y_mm: float = DEFAULT_PIXEL_SPACING_Y_MM,
        spacing_x_mm: float = DEFAULT_PIXEL_SPACING_X_MM,
    ) -> np.ndarray:
        size = 8
        baseline_features = extract_features(normalized, region, key, spacing_y_mm, spacing_x_mm)
        height, width = normalized.shape
        replacement = float(np.median(normalized))
        occluded_rows = []
        for row in range(size):
            y0, y1 = row * height // size, (row + 1) * height // size
            for column in range(size):
                x0, x1 = column * width // size, (column + 1) * width // size
                occluded = normalized.copy()
                occluded[y0:y1, x0:x1] = replacement
                occluded_rows.append(
                    extract_features(occluded, region, key, spacing_y_mm, spacing_x_mm)
                )
        probabilities = self._probabilities(key, np.stack([baseline_features, *occluded_rows]))
        baseline = probabilities[0]
        heatmap = np.abs(baseline - np.asarray(probabilities[1:], dtype=np.float32)).reshape(size, size)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max() + 1e-8
        if region == LEFT_HIP:
            heatmap = np.fliplr(heatmap).copy()
        return heatmap
