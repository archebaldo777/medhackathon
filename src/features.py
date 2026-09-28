from __future__ import annotations

import math

import numpy as np
from PIL import Image

from .constants import (
    AXIS_TOLERANCE_DEG,
    DEFAULT_PIXEL_SPACING_X_MM,
    DEFAULT_PIXEL_SPACING_Y_MM,
    LEFT_HIP,
)


# Every acceptance criterion in the ТЗ is a statement about physical geometry or
# about a local hyperdense object, so each head is given the descriptor block
# built for its own criterion instead of one shared appearance vector. Sharing a
# 570-dimensional appearance vector across all seven heads forced heavy
# regularisation that buried the few informative directions: the measured spine
# tilt alone separates axis deviation better (AUC 0.80) than the whole shared
# vector did (AUC 0.56).
TARGET_BLOCKS: dict[str, tuple[str, ...]] = {
    "spine_quality": ("axis", "framing", "artifact"),
    "spine_positioning": ("framing",),
    "spine_axis_deviation": ("axis",),
    "foreign_object_or_artifact": ("artifact",),
    "hip_quality": ("framing", "hip_shape"),
    "hip_positioning_or_rotation": ("hip_shape",),
    "hip_roi_incorrect": ("framing",),
}


def _resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    pil = Image.fromarray(np.clip(image * 255.0, 0, 255).astype(np.uint8))
    return np.asarray(pil.resize((width, height), Image.Resampling.BILINEAR), dtype=np.float32) / 255.0


def canonical_image(image: np.ndarray, region: str) -> np.ndarray:
    """Mirror left hips into the right-hip orientation so shape features align."""
    if region == LEFT_HIP:
        return np.fliplr(image).copy()
    return image


def _intensity_summary(image: np.ndarray) -> list[float]:
    quantiles = np.quantile(image, [0.05, 0.25, 0.5, 0.75, 0.9, 0.97, 0.995])
    return [float(image.mean()), float(image.std()), *(float(v) for v in quantiles)]


def _frame(image: np.ndarray, spacing_y_mm: float, spacing_x_mm: float) -> list[float]:
    """Physical size of the acquisition field; a truncated field is itself a finding."""
    height, width = image.shape
    return [
        float(height * spacing_y_mm),
        float(width * spacing_x_mm),
        float((height * spacing_y_mm) / max(width * spacing_x_mm, 1e-6)),
    ]


def _column_trace(image: np.ndarray, quantile: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Median column of the bright structure per row, with its row width."""
    height, width = image.shape
    mask = image > float(np.quantile(image, quantile))
    rows: list[float] = []
    centres: list[float] = []
    widths: list[float] = []
    step = max(1, height // 48)
    for y in range(int(height * 0.06), int(height * 0.94), step):
        columns = np.where(mask[y])[0]
        central = columns[(columns > width * 0.15) & (columns < width * 0.85)]
        if len(central) > 2:
            rows.append(float(y))
            centres.append(float(np.median(central)))
            widths.append(float(len(central)))
    return np.asarray(rows), np.asarray(centres), np.asarray(widths)


def _tilt_degrees(
    rows: np.ndarray, centres: np.ndarray, spacing_y_mm: float, spacing_x_mm: float
) -> tuple[float, float]:
    """Fit the column and return its tilt in physical degrees plus the fit error.

    The scanner pixels are anisotropic (1.05 mm by row, 0.60 mm by column), so an
    angle measured in pixel space is wrong by a factor of about 1.75. The slope is
    therefore converted to millimetres before taking the arctangent.
    """
    if len(rows) < 4:
        return 0.0, 0.0
    slope, intercept = np.polyfit(rows, centres, 1)
    residual = centres - (slope * rows + intercept)
    angle = math.degrees(math.atan(slope * spacing_x_mm / spacing_y_mm))
    return float(angle), float(np.sqrt(np.mean(residual**2)) * spacing_x_mm)


def axis_block(image: np.ndarray, spacing_y_mm: float, spacing_x_mm: float) -> np.ndarray:
    """ТЗ 2.3: the lumbar axis may deviate from vertical by at most 5 degrees."""
    rows, centres, widths = _column_trace(image, 0.78)
    angle, residual_mm = _tilt_degrees(rows, centres, spacing_y_mm, spacing_x_mm)
    if len(rows) >= 12:
        cut = len(rows) // 3
        thirds = [
            _tilt_degrees(rows[segment], centres[segment], spacing_y_mm, spacing_x_mm)[0]
            for segment in (slice(0, cut), slice(cut, 2 * cut), slice(2 * cut, None))
        ]
    else:
        thirds = [angle, angle, angle]
    drift_mm = float((centres.max() - centres.min()) * spacing_x_mm) if len(centres) else 0.0
    values = [
        angle,
        abs(angle),
        max(0.0, abs(angle) - AXIS_TOLERANCE_DEG),
        residual_mm,
        drift_mm,
        *thirds,
        abs(thirds[0] - thirds[2]),
        float(np.mean(widths) * spacing_x_mm) if len(widths) else 0.0,
        float(np.std(widths) * spacing_x_mm) if len(widths) else 0.0,
    ]
    return np.asarray(
        values + _intensity_summary(image) + _frame(image, spacing_y_mm, spacing_x_mm),
        dtype=np.float32,
    )


def _bone_box(image: np.ndarray, quantile: float) -> tuple[float, float, float, float] | None:
    mask = image > float(np.quantile(image, quantile))
    rows, columns = np.where(mask)
    if len(columns) < 10:
        return None
    return float(rows.min()), float(rows.max()), float(columns.min()), float(columns.max())


def framing_block(image: np.ndarray, spacing_y_mm: float, spacing_x_mm: float) -> np.ndarray:
    """How much field surrounds the bone, in millimetres.

    ТЗ 2.3 and figure 6 define correct framing in centimetres — 3 cm above and
    below the region of interest and 2 cm from the lateral edge — so the margins
    are kept in physical units. A margin expressed as a fraction of the frame,
    which is what a scale-free appearance vector carries, cannot express them.
    """
    height, width = image.shape
    values: list[float] = []
    for quantile in (0.80, 0.90):
        box = _bone_box(image, quantile)
        if box is None:
            values.extend([0.0] * 8)
            continue
        top, bottom, left, right = box
        top_mm = top * spacing_y_mm
        bottom_mm = (height - 1 - bottom) * spacing_y_mm
        left_mm = left * spacing_x_mm
        right_mm = (width - 1 - right) * spacing_x_mm
        values.extend(
            [
                top_mm,
                bottom_mm,
                left_mm,
                right_mm,
                (bottom - top) * spacing_y_mm,
                (right - left) * spacing_x_mm,
                min(left_mm, right_mm),
                min(top_mm, bottom_mm),
            ]
        )
    # Vertical distribution of bone mass: a truncated field ends abruptly.
    profile = image.mean(axis=1)
    values.extend(float(band.mean()) for band in np.array_split(profile, 6))
    values.append(float(profile[: max(height // 12, 1)].mean()))
    values.append(float(profile[-max(height // 12, 1) :].mean()))
    column = image.mean(axis=0)
    values.append(float(column[: max(width // 12, 1)].mean()))
    values.append(float(column[-max(width // 12, 1) :].mean()))
    return np.asarray(
        values + _intensity_summary(image) + _frame(image, spacing_y_mm, spacing_x_mm),
        dtype=np.float32,
    )


def artifact_block(image: np.ndarray, spacing_y_mm: float, spacing_x_mm: float) -> np.ndarray:
    """ТЗ 2.3 and figure 3: hyperdense foreign bodies, typically clothing metal.

    A foreign body is a sharp local spike rather than a broad bone, so the block
    compares the image with its own smoothed version and records where the bright
    tail sits — the examples in the ТЗ sit at the edges of the field.
    """
    height, width = image.shape
    small = _resize(image, max(width // 8, 4), max(height // 8, 4))
    smooth = (
        np.asarray(
            Image.fromarray((small * 255).astype(np.uint8)).resize(
                (width, height), Image.Resampling.BILINEAR
            ),
            dtype=np.float32,
        )
        / 255.0
    )
    tophat = image - smooth

    p50, p90, p99, p999 = (float(v) for v in np.quantile(image, [0.5, 0.9, 0.99, 0.999]))
    bright = image > p99
    band_height, band_width = max(height // 8, 1), max(width // 8, 1)
    edges = [
        float(image[:band_height, :].mean()),
        float(image[-band_height:, :].mean()),
        float(image[:, :band_width].mean()),
        float(image[:, -band_width:].mean()),
    ]
    bright_edges = [
        float(bright[:band_height, :].mean()),
        float(bright[-band_height:, :].mean()),
        float(bright[:, :band_width].mean()),
        float(bright[:, -band_width:].mean()),
    ]
    values = [
        float(tophat.max()),
        float(np.quantile(tophat, 0.999)),
        float(np.quantile(tophat, 0.99)),
        float((tophat > 0.15).mean()),
        float((tophat > 0.25).mean()),
        p99 - p90,
        p999 - p99,
        p90 - p50,
        float(bright.mean()),
        *edges,
        *bright_edges,
        float(max(edges) - min(edges)),
    ]
    return np.asarray(
        values + _intensity_summary(image) + _frame(image, spacing_y_mm, spacing_x_mm),
        dtype=np.float32,
    )


def hip_shape_block(image: np.ndarray, spacing_y_mm: float, spacing_x_mm: float) -> np.ndarray:
    """ТЗ 2.3 and figures 4-5: trochanter presentation and femoral rotation.

    Rotation is judged by how far the lesser trochanter deforms the medial
    outline, so the medial boundary is traced and compared with its own smooth
    trend. Images arrive canonicalised to the right-hip orientation.
    """
    height, width = image.shape
    mask = image > float(np.quantile(image, 0.80))
    rows: list[float] = []
    medial: list[float] = []
    lateral: list[float] = []
    widths: list[float] = []
    for y in range(int(height * 0.35), int(height * 0.97), max(1, height // 60)):
        columns = np.where(mask[y])[0]
        if len(columns) < 3:
            continue
        rows.append(float(y))
        medial.append(float(columns.min()))
        lateral.append(float(columns.max()))
        widths.append(float(columns.max() - columns.min()))

    if len(rows) < 6:
        base = [0.0] * 12
    else:
        row_array = np.asarray(rows)
        medial_array = np.asarray(medial)
        lateral_array = np.asarray(lateral)
        width_array = np.asarray(widths)
        slope, intercept = np.polyfit(row_array, medial_array, 1)
        deviation = (slope * row_array + intercept - medial_array) * spacing_x_mm
        shaft_angle = math.degrees(math.atan(slope * spacing_x_mm / spacing_y_mm))
        base = [
            float(deviation.max()),
            float(deviation.std()),
            float(np.quantile(deviation, 0.9)),
            float(row_array[int(np.argmax(deviation))] / max(height - 1, 1)),
            shaft_angle,
            abs(shaft_angle),
            float(width_array.min() * spacing_x_mm),
            float(width_array.max() * spacing_x_mm),
            float(width_array.mean() * spacing_x_mm),
            float(width_array.std() * spacing_x_mm),
            float((lateral_array.max() - lateral_array.min()) * spacing_x_mm),
            float((medial_array.max() - medial_array.min()) * spacing_x_mm),
        ]
    grid = _resize(image, 8, 8).ravel().tolist()
    return np.asarray(
        base + grid + _intensity_summary(image) + _frame(image, spacing_y_mm, spacing_x_mm),
        dtype=np.float32,
    )


BLOCK_BUILDERS = {
    "axis": axis_block,
    "framing": framing_block,
    "artifact": artifact_block,
    "hip_shape": hip_shape_block,
}


def extract_features(
    normalized: np.ndarray,
    region: str,
    target: str,
    spacing_y_mm: float = DEFAULT_PIXEL_SPACING_Y_MM,
    spacing_x_mm: float = DEFAULT_PIXEL_SPACING_X_MM,
) -> np.ndarray:
    """Build the descriptor one head needs, in physical units."""
    blocks = TARGET_BLOCKS.get(target)
    if blocks is None:
        raise ValueError(f"Неизвестная цель модели: {target}")
    image = canonical_image(normalized, region)
    spacing_y_mm = float(spacing_y_mm or DEFAULT_PIXEL_SPACING_Y_MM)
    spacing_x_mm = float(spacing_x_mm or DEFAULT_PIXEL_SPACING_X_MM)
    parts = [BLOCK_BUILDERS[block](image, spacing_y_mm, spacing_x_mm) for block in blocks]
    return np.concatenate(parts).astype(np.float32)


def spine_axis_line(
    normalized: np.ndarray,
    spacing_y_mm: float = DEFAULT_PIXEL_SPACING_Y_MM,
    spacing_x_mm: float = DEFAULT_PIXEL_SPACING_X_MM,
) -> dict[str, float] | None:
    """The fitted lumbar axis in image coordinates plus its physical tilt.

    This is the same trace the axis descriptor uses, so the line drawn on the
    visualisation and the angle written to the report are the ones the model saw.
    """
    rows, centres, _ = _column_trace(normalized, 0.78)
    if len(rows) < 4:
        return None
    slope, intercept = np.polyfit(rows, centres, 1)
    angle, residual_mm = _tilt_degrees(rows, centres, spacing_y_mm, spacing_x_mm)
    return {
        "row_start": float(rows[0]),
        "row_end": float(rows[-1]),
        "column_start": float(slope * rows[0] + intercept),
        "column_end": float(slope * rows[-1] + intercept),
        "angle_deg": angle,
        "residual_mm": residual_mm,
    }


def measure_geometry(
    normalized: np.ndarray,
    region: str,
    spacing_y_mm: float = DEFAULT_PIXEL_SPACING_Y_MM,
    spacing_x_mm: float = DEFAULT_PIXEL_SPACING_X_MM,
) -> dict[str, float | bool]:
    """Physical measurements reported next to the decision, for the specialist."""
    height, width = normalized.shape
    spacing_y_mm = float(spacing_y_mm or DEFAULT_PIXEL_SPACING_Y_MM)
    spacing_x_mm = float(spacing_x_mm or DEFAULT_PIXEL_SPACING_X_MM)
    measurements: dict[str, float | bool] = {
        "field_height_mm": round(height * spacing_y_mm, 1),
        "field_width_mm": round(width * spacing_x_mm, 1),
    }
    if region == "lumbar_spine":
        axis = spine_axis_line(normalized, spacing_y_mm, spacing_x_mm)
        if axis is not None:
            measurements["axis_angle_deg"] = round(axis["angle_deg"], 2)
            measurements["axis_tolerance_deg"] = AXIS_TOLERANCE_DEG
            measurements["axis_within_tolerance"] = abs(axis["angle_deg"]) <= AXIS_TOLERANCE_DEG
    return measurements


def feature_size(target: str) -> int:
    probe = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
    return int(extract_features(probe, "lumbar_spine", target).shape[0])


def feature_sizes() -> dict[str, int]:
    return {target: feature_size(target) for target in TARGET_BLOCKS}
