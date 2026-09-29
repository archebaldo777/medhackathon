from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .constants import (
    ANATOMICAL_REGION_NAMES_RU,
    LEFT_HIP,
    RIGHT_HIP,
    AXIS_TOLERANCE_DEG,
    HIP_INFERIOR_MARGIN_CM,
    HIP_LATERAL_MARGIN_CM,
    HIP_SUPERIOR_MARGIN_CM,
    REPORT_COLUMNS,
    SPINE,
)
from .dicom_io import (
    anatomical_region,
    distance_cm_to_pixels,
    normalize_pixels,
    read_dicom,
    read_identifiers,
)
from .features import measure_geometry
from .markup import SCOLIOSIS_DEVIATION_MM, build_markup
from .model import DEFAULT_MODEL_PATH, ModelResult, QualityModel
from .projection import LATERAL, detect_projection, laterality_from_tags


# A negative result deserves a second look when its probability sits just under
# the head's own operating point. The margin is relative to that threshold
# because the heads are calibrated far away from 0.5 — a fixed band around 0.5
# would never trigger for a threshold of 0.09 and would mark clearly negative
# images for one of 1.0.
BORDERLINE_RELATIVE_MARGIN = 0.15


def _failed_checks(markup: dict[str, Any]) -> str:
    failed = []
    for check in markup.get("checks", []):
        if check.get("ok") is False:
            if check.get("measured") is not None:
                failed.append(
                    f"{check['criterion'].lower()}: {check['measured']:.0f} мм при норме "
                    f"не менее {check['required']:.0f} мм"
                )
            else:
                failed.append(check["criterion"].lower() + ": нет")
    return "; ".join(failed)


def positioning_recommendations(
    predicted: ModelResult, measurements: dict[str, Any], markup: dict[str, Any]
) -> list[dict[str, Any]]:
    """What the technologist should change at the next acquisition.

    These are recommendations about patient positioning and the scan field. They
    are not the markup correction of ТЗ 2.6: the markup is the densitometer's
    ROI placement, which the service reconstructs in ``markup`` and the
    specialist confirms or edits. A recommendation is emitted only for a
    violation the model reported and carries the measured value when the
    service measures that criterion itself.
    """
    items: list[dict[str, Any]] = []
    angle = measurements.get("axis_angle_deg")
    if "spine_axis_deviation" in predicted.violation_keys and angle is not None:
        direction = "по часовой стрелке" if angle > 0 else "против часовой стрелки"
        items.append(
            {
                "criterion": "Не выравнена ось позвоночника",
                "measured": f"наклон {angle:+.1f}°, допуск ±{AXIS_TOLERANCE_DEG:.0f}°",
                "action": (
                    f"Выровнять ось пациента по оси стола: развернуть на {abs(angle):.1f}° "
                    f"{direction}; эталонная вертикаль показана пунктиром на визуализации."
                ),
            }
        )
    curve = (markup.get("measurements") or {}).get("curve_deviation_mm")
    axis_item = next((item for item in items if item["criterion"] == "Не выравнена ось позвоночника"), None)
    if axis_item and curve is not None and curve >= SCOLIOSIS_DEVIATION_MM:
        axis_item["measured"] += (
            f"; ось изогнута дугой до {curve:.0f} мм — вероятен сколиоз, который эксперты "
            "не считают нарушением оси: проверьте, не он ли вызвал срабатывание"
        )
    failed = _failed_checks(markup)
    for key, title, action in (
        (
            "spine_positioning",
            "Некорректная укладка",
            "Сместить поле сканирования: снизу должны попасть верхние края "
            "подвздошных костей, сверху — половина тела Th12.",
        ),
        (
            "foreign_object_or_artifact",
            "Присутствуют посторонние предметы",
            "Убрать металлические предметы одежды из зоны сканирования и повторить "
            "исследование; область с наибольшим вкладом выделена на тепловой карте.",
        ),
        (
            "hip_positioning_or_rotation",
            "Некорректная укладка",
            "Проверить ротацию конечности по малому вертелу и видимость большого "
            "вертела, шейки бедра и седалищной кости.",
        ),
        (
            "hip_roi_incorrect",
            "Некорректная область интереса",
            "Расширить поле: не менее 3 см выше и ниже области интереса и 2 см от "
            "бокового края.",
        ),
    ):
        if key in predicted.violation_keys:
            measured = f"оценка модели {predicted.probabilities.get(key, 0.0):.2f}"
            if key in {"spine_positioning", "hip_roi_incorrect"} and failed:
                measured += f"; по разметке — {failed}"
            items.append({"criterion": title, "measured": measured, "action": action})
    return items


def is_borderline(probability: float, threshold: float) -> bool:
    """Report whether a sub-threshold probability is close to the decision."""
    if not 0.0 < threshold <= 1.0:
        return False
    return threshold * (1.0 - BORDERLINE_RELATIVE_MARGIN) <= probability < threshold


@dataclass
class AnalysisResult:
    path_to_study: str
    study_uid: str = ""
    image_uid: str = ""
    anatomical_region: str = "unknown"
    projection: str = ""
    projection_code: str = ""
    projection_source: str = ""
    laterality_source: str = ""
    quality_class: int | str = ""
    violation_type: str = ""
    processing_status: str = "Failure"
    time_of_processing: float = 0.0
    quality_prob: float | None = None
    violations: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""
    model_version: str = ""
    review_priority: str = ""
    recommendation: str = ""
    decision_threshold: float | None = None
    pixel_spacing_y_mm: float | None = None
    pixel_spacing_x_mm: float | None = None
    pixel_spacing_source: str = ""
    geometry_thresholds_px: dict[str, float] = field(default_factory=dict)
    measurements: dict[str, Any] = field(default_factory=dict)
    head_scores: dict[str, dict[str, float]] = field(default_factory=dict)
    recommendations: list[dict[str, Any]] = field(default_factory=list)
    markup: dict[str, Any] = field(default_factory=dict)
    review: dict[str, Any] = field(default_factory=dict)
    artifacts: dict[str, str] = field(default_factory=dict)

    def report_row(self) -> dict[str, Any]:
        values = asdict(self)
        return {column: values[column] for column in REPORT_COLUMNS}

    def api_dict(self) -> dict[str, Any]:
        return asdict(self)


class Analyzer:
    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ):
        self.model = QualityModel(model_path)

    @staticmethod
    def _dxa_result(
        shown_path: str,
        dicom: Any,
        region: str,
        predicted: ModelResult,
        normalized: Any = None,
    ) -> AnalysisResult:
        borderline = is_borderline(predicted.quality_probability, predicted.decision_threshold)
        tags = getattr(dicom, "tags", None) or {}
        projection = detect_projection(tags)
        lateral = projection.code == LATERAL
        side = region
        laterality_source = ""
        if region in (RIGHT_HIP, LEFT_HIP):
            tagged_side = laterality_from_tags(tags)
            side = tagged_side or region
            laterality_source = "dicom:Laterality" if tagged_side else "image"
        measurements = (
            measure_geometry(
                normalized, region, dicom.pixel_spacing_y_mm, dicom.pixel_spacing_x_mm
            )
            if normalized is not None
            else {}
        )
        markup = (
            build_markup(normalized, region, dicom.pixel_spacing_y_mm, dicom.pixel_spacing_x_mm).as_dict()
            if normalized is not None
            else {}
        )
        if markup:
            measurements.update(markup.get("measurements") or {})
            markup["image_shape"] = [int(normalized.shape[0]), int(normalized.shape[1])]
            markup["pixel_spacing_y_mm"] = float(dicom.pixel_spacing_y_mm)
            markup["pixel_spacing_x_mm"] = float(dicom.pixel_spacing_x_mm)
            markup["status"] = "proposed"
        head_scores = {
            key: {
                "probability": round(float(value), 6),
                "threshold": round(float((predicted.head_thresholds or {}).get(key, 0.5)), 6),
            }
            for key, value in predicted.probabilities.items()
        }
        return AnalysisResult(
            path_to_study=shown_path,
            study_uid=dicom.study_uid,
            image_uid=dicom.image_uid,
            anatomical_region=ANATOMICAL_REGION_NAMES_RU[side],
            projection=projection.name_ru,
            projection_code=projection.code,
            projection_source=projection.source,
            laterality_source=laterality_source,
            quality_class=predicted.quality_class,
            violation_type=";".join(predicted.violation_labels),
            processing_status="Success",
            quality_prob=round(predicted.quality_probability, 6),
            violations=predicted.explanations,
            model_version=predicted.model_version,
            decision_threshold=round(predicted.decision_threshold, 6),
            pixel_spacing_y_mm=dicom.pixel_spacing_y_mm,
            pixel_spacing_x_mm=dicom.pixel_spacing_x_mm,
            pixel_spacing_source=dicom.pixel_spacing_source,
            geometry_thresholds_px={} if region == SPINE else {
                "above_greater_trochanter_3cm_y": round(
                    distance_cm_to_pixels(HIP_SUPERIOR_MARGIN_CM, dicom.pixel_spacing_y_mm),
                    6,
                ),
                "below_region_of_interest_3cm_y": round(
                    distance_cm_to_pixels(HIP_INFERIOR_MARGIN_CM, dicom.pixel_spacing_y_mm),
                    6,
                ),
                "from_lateral_edge_2cm_x": round(
                    distance_cm_to_pixels(HIP_LATERAL_MARGIN_CM, dicom.pixel_spacing_x_mm),
                    6,
                ),
            },
            measurements=measurements,
            head_scores=head_scores,
            recommendations=positioning_recommendations(predicted, measurements, markup),
            markup=markup,
            review_priority=(
                "high"
                if predicted.quality_class == 1 or lateral
                else "medium" if borderline else "low"
            ),
            recommendation=(
                "Боковая проекция: модель обучена на прямых снимках, результат требует обязательной "
                "проверки специалистом."
                if lateral
                else "Проверить укладку и подтвердить разметку области измерения до клинической интерпретации."
                if predicted.quality_class == 1
                else "Пограничная уверенность: рекомендована ручная проверка."
                if borderline
                else "Автоматическая проверка пройдена; стандартный контроль специалиста сохраняется."
            ),
        )

    def analyze_many(
        self, items: list[tuple[str | Path, str | None]]
    ) -> list[tuple[AnalysisResult, Any, ModelResult | None]]:
        """Analyze a batch, pooling DICOM frames by study and internal region.

        ``time_of_processing`` of each image is its own work only: reading and
        preprocessing the file, an equal share of its group's model inference and
        building its result. Artefacts (SR, markup, visualisations) are added by
        the batch writer.
        """
        outputs: list[tuple[AnalysisResult, Any, ModelResult | None] | None] = [
            None
        ] * len(items)
        groups: dict[tuple[str, str], list[tuple[int, Any, Any, str, float]]] = {}
        for index, (path, display_path) in enumerate(items):
            source = Path(path)
            shown_path = display_path or str(source)
            started = time.perf_counter()
            try:
                dicom = read_dicom(source)
                normalized = normalize_pixels(dicom.pixels)
                # Only the normalised image is used from here on; releasing the raw
                # pixels halves the memory held while a large batch is grouped.
                dicom.pixels = np.empty(0, dtype=np.float32)
                region = anatomical_region(normalized)
                study_key = dicom.study_uid or str(source)
                groups.setdefault((study_key, region), []).append(
                    (index, dicom, normalized, shown_path, time.perf_counter() - started)
                )
            except Exception as exc:
                study_uid, image_uid = read_identifiers(source)
                result = AnalysisResult(
                    path_to_study=shown_path,
                    study_uid=study_uid,
                    image_uid=image_uid,
                    error=str(exc),
                    time_of_processing=round(time.perf_counter() - started, 4),
                )
                outputs[index] = (result, None, None)

        for (_, region), entries in groups.items():
            reference = entries[0][1]
            group_started = time.perf_counter()
            try:
                predictions = self.model.predict_group(
                    [entry[2] for entry in entries],
                    region,
                    reference.pixel_spacing_y_mm,
                    reference.pixel_spacing_x_mm,
                )
            except Exception as exc:  # a failing group must not abort the batch
                share = (time.perf_counter() - group_started) / len(entries)
                for index, dicom, _, shown_path, read_seconds in entries:
                    result = AnalysisResult(
                        path_to_study=shown_path,
                        study_uid=dicom.study_uid,
                        image_uid=dicom.image_uid,
                        error=f"Ошибка анализа: {exc}",
                        time_of_processing=round(read_seconds + share, 4),
                    )
                    outputs[index] = (result, None, None)
                continue
            share = (time.perf_counter() - group_started) / len(entries)
            for entry, predicted in zip(entries, predictions):
                index, dicom, normalized, shown_path, read_seconds = entry
                built = time.perf_counter()
                try:
                    result = self._dxa_result(shown_path, dicom, region, predicted, normalized)
                except Exception as exc:
                    result = AnalysisResult(
                        path_to_study=shown_path,
                        study_uid=dicom.study_uid,
                        image_uid=dicom.image_uid,
                        error=f"Ошибка формирования результата: {exc}",
                    )
                    normalized, predicted = None, None
                result.time_of_processing = round(
                    read_seconds + share + time.perf_counter() - built, 4
                )
                outputs[index] = (result, normalized, predicted)
        if any(output is None for output in outputs):
            raise RuntimeError("Не все входные файлы получили результат анализа")
        return [output for output in outputs if output is not None]

    def analyze(
        self, path: str | Path, display_path: str | None = None
    ) -> tuple[AnalysisResult, Any, ModelResult | None]:
        started = time.perf_counter()
        source = Path(path)
        shown_path = display_path or str(source)
        try:
            dicom = read_dicom(source)
            normalized = normalize_pixels(dicom.pixels)
            region = anatomical_region(normalized)
            predicted = self.model.predict(
                normalized, region, dicom.pixel_spacing_y_mm, dicom.pixel_spacing_x_mm
            )
            result = self._dxa_result(shown_path, dicom, region, predicted, normalized)
            return result, normalized, predicted
        except Exception as exc:  # one malformed image must not abort a batch
            study_uid, image_uid = read_identifiers(source)
            result = AnalysisResult(
                path_to_study=shown_path, study_uid=study_uid, image_uid=image_uid, error=str(exc)
            )
            return result, None, None
        finally:
            elapsed = time.perf_counter() - started
            if "result" in locals():
                result.time_of_processing = round(elapsed, 4)
