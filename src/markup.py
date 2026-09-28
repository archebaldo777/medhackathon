"""Automatic anatomical markup of a DXA image and its geometric checks.

«Разметка анатомических структур и областей измерения» in the ТЗ is the markup
the densitometer draws on the image while the study is processed: the
intervertebral lines that delimit L1–L4 for the spine, the femoral neck box and
the total-hip region of interest for the hip. The specialist corrects it in the
densitometer software. The DICOM export supplied for the task carries the image
only (8-bit MONOCHROME2, no overlay planes, no presentation state), so the
densitometer markup itself cannot be read. The service therefore reconstructs
the same markup from the image, checks it against the ТЗ criteria and offers it
to the specialist, who confirms it, moves its points or rejects it.

All geometry is computed on an isotropic 1 mm grid, because the scanner pixel is
1.05 mm by row and 0.60 mm by column; results are returned both in millimetres
and in source-image pixels. The expert spreadsheet shipped with the task labels
quality violations for training; it contains no markup and is not used here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image
from scipy import ndimage, signal

from .constants import (
    DEFAULT_PIXEL_SPACING_X_MM,
    DEFAULT_PIXEL_SPACING_Y_MM,
    HIP_INFERIOR_MARGIN_CM,
    HIP_LATERAL_MARGIN_CM,
    HIP_SUPERIOR_MARGIN_CM,
    LEFT_HIP,
    SPINE,
)

SPINE_DISC_LABELS = ("T12/L1", "L1/L2", "L2/L3", "L3/L4", "L4/L5")
VERTEBRA_LABELS = ("L1", "L2", "L3", "L4")
CREST_LABEL = "Гребни подвздошных костей"
HIP_ROI_LABEL = "Область интереса (проксимальный отдел бедра)"
NECK_LABEL = "Шейка бедра"
HEAD_LABEL = "Головка бедра"
SPINE_EDGE_LABELS = ("Контур позвоночника слева", "Контур позвоночника справа")
BONE_EDGE_LABEL = "Контур кости"
TROCHANTER_LINE_LABEL = "Линия вертела"
WARD_LABEL = "Треугольник Варда"
# Shapes that come from the image itself and stay as they are when the
# specialist edits the editable ROI elements.
SCOLIOSIS_DEVIATION_MM = 6.0
PUBLIC_METRICS = ("curve_deviation_mm",)
STATIC_LABELS = {HEAD_LABEL, BONE_EDGE_LABEL, *SPINE_EDGE_LABELS}

# Drawing styles follow the densitometer screen: yellow bone edge, cyan ROIs.
STYLE_EDGE = "edge"
STYLE_ROI = "roi"
STYLE_LANDMARK = "landmark"


@dataclass
class Markup:
    """Markup shapes in source-image pixels plus the ТЗ checks derived from them.

    ``shapes`` is the editable part: each shape is a named polyline or polygon.
    Vertebra boxes are derived from the intervertebral lines and are rebuilt
    whenever the specialist moves a line, so only the lines are editable.
    """

    region: str
    shapes: list[dict[str, Any]] = field(default_factory=list)
    derived: list[dict[str, Any]] = field(default_factory=list)
    checks: list[dict[str, Any]] = field(default_factory=list)
    confidence: str = "low"
    notes: list[str] = field(default_factory=list)
    source: str = "automatic"
    metrics: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "region": self.region,
            "source": self.source,
            "confidence": self.confidence,
            "shapes": self.shapes,
            "derived": self.derived,
            "checks": self.checks,
            "notes": self.notes,
            "measurements": {
                key: self.metrics[key] for key in PUBLIC_METRICS if key in self.metrics
            },
        }


class _Grid:
    """Maps between the isotropic millimetre grid and source pixels."""

    def __init__(self, shape: tuple[int, int], spacing_y_mm: float, spacing_x_mm: float, mirror: bool):
        self.rows, self.columns = shape
        self.height_mm = max(8, int(round(self.rows * spacing_y_mm)))
        self.width_mm = max(8, int(round(self.columns * spacing_x_mm)))
        self.mirror = mirror

    def resample(self, image: np.ndarray) -> np.ndarray:
        source = np.fliplr(image) if self.mirror else image
        pil = Image.fromarray((np.clip(source, 0.0, 1.0) * 255).astype(np.uint8))
        resized = pil.resize((self.width_mm, self.height_mm), Image.Resampling.BILINEAR)
        return np.asarray(resized, dtype=np.float32) / 255.0

    def to_pixels(self, x_mm: float, y_mm: float) -> list[float]:
        x = float(x_mm) * self.columns / self.width_mm
        y = float(y_mm) * self.rows / self.height_mm
        if self.mirror:
            x = self.columns - 1 - x
        return [round(x, 2), round(y, 2)]

    def to_mm(self, x_px: float, y_px: float) -> tuple[float, float]:
        x = self.columns - 1 - float(x_px) if self.mirror else float(x_px)
        return x * self.width_mm / self.columns, float(y_px) * self.height_mm / self.rows


def _otsu(values: np.ndarray) -> float:
    hist, edges = np.histogram(values, bins=64, range=(0.0, 1.0))
    centres = (edges[:-1] + edges[1:]) / 2
    weight = np.cumsum(hist)
    mass = np.cumsum(hist * centres)
    total, total_mass = weight[-1], mass[-1]
    between = (total_mass * weight - mass * total) ** 2 / np.maximum(weight * (total - weight), 1e-9)
    return float(centres[int(np.argmax(between))])


def _trace_boundary(mask: np.ndarray, step: int = 3) -> list[tuple[float, float]]:
    """Ordered outer boundary of the largest component (Moore neighbour tracing)."""
    labels, count = ndimage.label(mask)
    if count == 0:
        return []
    sizes = ndimage.sum(mask, labels, range(1, count + 1))
    component = np.pad(labels == int(np.argmax(sizes)) + 1, 1)
    ys, xs = np.nonzero(component)
    start = (int(ys[0]), int(xs[0]))
    # Clockwise neighbours starting from the west.
    ring = [(0, -1), (-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1)]
    boundary = [start]
    current = start
    backtrack = (start[0], start[1] - 1)
    limit = int(component.sum()) * 2 + 16
    for _ in range(limit):
        offset = (backtrack[0] - current[0], backtrack[1] - current[1])
        k = ring.index(offset)
        found = None
        for turn in range(1, 9):
            dy, dx = ring[(k + turn) % 8]
            candidate = (current[0] + dy, current[1] + dx)
            if component[candidate]:
                found = candidate
                break
            backtrack = candidate
        if found is None or found == start:
            break
        current = found
        boundary.append(current)
    points = [(float(x - 1), float(y - 1)) for y, x in boundary[::step]]
    return points + points[:1]


# ---------------------------------------------------------------- spine ----


def _spine_centreline(smooth: np.ndarray) -> np.ndarray:
    height, width = smooth.shape
    band = ndimage.uniform_filter1d(smooth, size=30, axis=1)
    rows = np.arange(int(height * 0.03), int(height * 0.97))
    low, high = int(width * 0.2), int(width * 0.8)
    centres = np.asarray([low + int(np.argmax(band[y, low:high])) for y in rows], dtype=float)
    coef = np.polyfit(rows, centres, 2)
    for _ in range(3):
        residual = centres - np.polyval(coef, rows)
        keep = np.abs(residual) < max(6.0, 2.5 * float(np.std(residual)))
        coef = np.polyfit(rows[keep], centres[keep], 2)
    return np.polyval(coef, np.arange(height))


def _crest_level(smooth: np.ndarray, centre: np.ndarray) -> tuple[int | None, float]:
    """Row where the iliac wings appear lateral to the column, and its contrast."""
    height, width = smooth.shape
    lateral = np.zeros(height)
    for y in range(height):
        c = centre[y]
        left = smooth[y, max(0, int(c - 80)) : max(0, int(c - 35))]
        right = smooth[y, min(width, int(c + 35)) : min(width, int(c + 80))]
        values = [part.mean() for part in (left, right) if part.size]
        lateral[y] = min(values) if len(values) == 2 else (values[0] if values else 0.0)
    middle = float(np.median(lateral[int(height * 0.3) : int(height * 0.6)]))
    contrast = float(np.percentile(lateral[int(height * 0.6) :], 90) - middle)
    if contrast < 0.08:
        return None, contrast
    threshold = middle + 0.5 * contrast
    rows = np.where(lateral > threshold)[0]
    rows = rows[rows > height * 0.6]
    for y in rows:
        if np.all(lateral[y : y + 6] > threshold):
            return int(y), contrast
    return (int(rows.min()) if rows.size else None), contrast


def _disc_chain(profile: np.ndarray, crest: int | None) -> list[int] | None:
    """Five intervertebral levels going up from L4/L5, spaced like lumbar vertebrae."""
    height = len(profile)
    candidates, _ = signal.find_peaks(-profile, distance=8)
    candidates = [int(c) for c in candidates if 8 < c < height - 4]
    depth = {c: float(-profile[c]) for c in candidates}
    anchors = [
        c
        for c in candidates
        if (crest is None and c > height * 0.45) or (crest is not None and abs(c - crest) <= 30)
    ]
    best: tuple[float, list[int]] | None = None
    for anchor in anchors:
        start = depth[anchor] - (0.004 * abs(anchor - crest) if crest is not None else 0.0)
        paths = [(start, [anchor])]
        for _ in range(4):
            extended = []
            for score, path in paths:
                last = path[-1]
                previous_gap = path[-2] - path[-1] if len(path) > 1 else None
                for candidate in candidates:
                    gap = last - candidate
                    if 22 <= gap <= 50:
                        penalty = 0.003 * abs(gap - previous_gap) if previous_gap else 0.0
                        extended.append((score + depth[candidate] - penalty, path + [candidate]))
            extended.sort(key=lambda item: -item[0])
            paths = extended[:30]
            if not paths:
                break
        for score, path in paths:
            if len(path) == 5 and (best is None or score > best[0]):
                best = (score, path)
    return best[1] if best else None


def _column_edges(smooth: np.ndarray, y: int, centre: float) -> tuple[int, int]:
    """Left and right bone edge of the column in one row, at half of the peak."""
    row = smooth[int(np.clip(y, 0, smooth.shape[0] - 1))]
    c = int(np.clip(round(centre), 0, len(row) - 1))
    peak = float(row[max(0, c - 5) : c + 6].mean())
    floor = float(np.percentile(row, 20))
    level = floor + 0.5 * (peak - floor)
    left = c
    while left > 0 and row[left] > level and c - left < 45:
        left -= 1
    right = c
    while right < len(row) - 1 and row[right] > level and right - c < 45:
        right += 1
    return left, right


def _curvature(smooth: np.ndarray, centre: np.ndarray) -> dict[str, float]:
    """How far the column bends away from a straight line (scoliosis), in mm.

    The ТЗ criterion is the alignment of the lumbar axis with the table. A
    scoliotic curve is a property of the patient, not a positioning error, and
    the experts marked such studies as correctly aligned. The midline between
    the two bone edges is smoothed and compared with its straight-line fit; the
    largest lateral deviation (the sagitta of the curve) separates the studies
    the experts annotated as scoliosis from the rest with ROC-AUC 0.73 and does
    not predict the axis violation itself (ROC-AUC 0.48).
    """
    height = smooth.shape[0]
    ys = np.arange(int(height * 0.08), int(height * 0.92))
    if len(ys) < 20:
        return {}
    mids = np.asarray([np.mean(_column_edges(smooth, y, centre[y])) for y in ys], dtype=float)
    mids = ndimage.gaussian_filter1d(ndimage.median_filter(mids, 15), 6)
    slope, intercept = np.polyfit(ys, mids, 1)
    deviation = float(np.max(np.abs(mids - (slope * ys + intercept))))
    return {
        "curve_deviation_mm": round(deviation, 1),
        "column_tilt_deg": round(math.degrees(math.atan(slope)), 2),
    }


def _column_half_width(smooth: np.ndarray, y: int, centre: float) -> float:
    row = smooth[int(np.clip(y, 0, smooth.shape[0] - 1))]
    c = int(np.clip(round(centre), 0, len(row) - 1))
    peak = float(row[max(0, c - 5) : c + 6].mean())
    floor = float(np.percentile(row, 20))
    level = floor + 0.5 * (peak - floor)
    left = c
    while left > 0 and row[left] > level and c - left < 45:
        left -= 1
    right = c
    while right < len(row) - 1 and row[right] > level and right - c < 45:
        right += 1
    return float(np.clip((right - left) / 2, 15, 30))


def spine_markup(
    normalized: np.ndarray,
    spacing_y_mm: float = DEFAULT_PIXEL_SPACING_Y_MM,
    spacing_x_mm: float = DEFAULT_PIXEL_SPACING_X_MM,
) -> Markup:
    grid = _Grid(normalized.shape, spacing_y_mm, spacing_x_mm, mirror=False)
    image = grid.resample(normalized)
    smooth = ndimage.gaussian_filter(image, 2.0)
    centre = _spine_centreline(smooth)
    height, width = image.shape
    core = np.asarray(
        [smooth[y, max(0, int(centre[y]) - 12) : int(centre[y]) + 13].mean() for y in range(height)]
    )
    profile = ndimage.gaussian_filter1d(core - ndimage.uniform_filter1d(core, 41), 1.5)
    crest, contrast = _crest_level(smooth, centre)
    chain = _disc_chain(profile, crest)
    estimated = False
    if not chain:
        # No convincing intervertebral minima: place the levels at the mean
        # lumbar spacing above the crest (or the lower part of the field), so the
        # specialist moves five lines instead of drawing them from scratch.
        anchor = crest if crest is not None else int(height * 0.8)
        chain = [int(anchor - 36 * k) for k in range(5) if anchor - 36 * k > 4]
        chain = chain if len(chain) == 5 else None
        estimated = chain is not None

    markup = Markup(region=SPINE)
    if crest is not None:
        markup.shapes.append(
            {
                "label": CREST_LABEL,
                "type": "POLYLINE",
                "style": STYLE_LANDMARK,
                "points": [grid.to_pixels(0, crest), grid.to_pixels(grid.width_mm - 1, crest)],
            }
        )
    if chain:
        # Like the densitometer, the L1–L4 boxes share one left and one right
        # border that clears the widest vertebra; ``chain`` runs from L4/L5
        # upwards while the labels run from the top down.
        mid = float(np.mean([centre[y] for y in chain]))
        half = max(_column_half_width(smooth, y, centre[y]) for y in chain) + 10
        left_x, right_x = max(0.0, mid - half), min(width - 1.0, mid + half)
        for label, y in zip(SPINE_DISC_LABELS, reversed(chain)):
            markup.shapes.append(
                {
                    "label": label,
                    "type": "POLYLINE",
                    "style": STYLE_ROI,
                    "points": [grid.to_pixels(left_x, y), grid.to_pixels(right_x, y)],
                }
            )
    # Bone edge along both sides of the column, drawn like the densitometer.
    rows = list(range(3, height - 3, 3))
    edges = np.asarray([_column_edges(smooth, y, centre[y]) for y in rows], dtype=float)
    if len(rows) > 8:
        for side, label in enumerate(SPINE_EDGE_LABELS):
            xs = ndimage.median_filter(edges[:, side], size=5)
            markup.derived.append(
                {
                    "label": label,
                    "type": "POLYLINE",
                    "style": STYLE_EDGE,
                    "points": [grid.to_pixels(x, y) for x, y in zip(xs, rows)],
                }
            )
    curve = _curvature(smooth, centre)
    markup.metrics = {
        "crest_found": float(crest is not None),
        "crest_contrast": round(contrast, 4),
        "crest_from_bottom_mm": float(height - crest) if crest is not None else 0.0,
        "chain_found": float(bool(chain) and not estimated),
        "top_level_mm": float(chain[-1]) if chain else 0.0,
        "mean_gap_mm": float(np.mean(np.diff(chain[::-1]))) if chain else 0.0,
        "height_mm": float(height),
        **curve,
    }
    markup.confidence = "high" if (chain and crest is not None and not estimated) else "low"
    if curve and curve["curve_deviation_mm"] >= SCOLIOSIS_DEVIATION_MM:
        markup.notes.append(
            f"Ось позвоночника изогнута дугой: отклонение от прямой до "
            f"{curve['curve_deviation_mm']:.0f} мм — вероятен сколиоз. Это особенность "
            "пациента, а не ошибка укладки: эксперты не считали сколиоз нарушением оси."
        )
    if estimated:
        markup.notes.append(
            "Межпозвонковые промежутки не выделены по изображению; уровни L1–L4 "
            "расставлены по средней высоте позвонка и требуют правки."
        )
    if crest is None:
        markup.notes.append(
            "Гребни подвздошных костей не найдены в поле сканирования; уровень L4/L5 "
            "определён без опоры на них и требует проверки."
        )
    if not chain:
        markup.notes.append("Межпозвонковые промежутки L1–L4 не выделены автоматически.")
    return finalize(markup, normalized.shape, spacing_y_mm, spacing_x_mm)


# ------------------------------------------------------------------ hip ----


def _hip_landmarks(image: np.ndarray) -> dict[str, Any]:
    height, width = image.shape
    smooth = ndimage.gaussian_filter(image, 1.5)
    wide = ndimage.uniform_filter1d(smooth, size=21, axis=1)
    rows = np.arange(int(height * 0.75), int(height * 0.96))
    centres, widths = [], []
    for y in rows:
        x = int(np.argmax(wide[y]))
        half = smooth[y] > 0.5 * (wide[y, x] + np.median(smooth[y]))
        left = x
        while left > 0 and half[left - 1]:
            left -= 1
        right = x
        while right < width - 1 and half[right + 1]:
            right += 1
        centres.append(x)
        widths.append(right - left + 1)
    centres_arr = np.asarray(centres, dtype=float)
    coef = np.polyfit(rows, centres_arr, 1)
    for _ in range(2):
        residual = centres_arr - np.polyval(coef, rows)
        keep = np.abs(residual) < max(3.0, 2 * float(np.std(residual)))
        coef = np.polyfit(rows[keep], centres_arr[keep], 1)
    # The shaft of the femur is close to vertical on a DXA hip scan.
    coef[0] = float(np.clip(coef[0], -0.27, 0.27))
    shaft_width = float(np.median(widths))
    # Bone threshold from the shaft itself: the trochanteric region is less
    # dense than the pelvis, and a global Otsu split tends to lose it.
    shaft_level = float(np.median([wide[y, int(c)] for y, c in zip(rows, centres_arr)]))
    background = float(np.percentile(smooth, 10))
    bone = smooth > background + 0.45 * (shaft_level - background)
    # A looser, hole-free mask follows the less dense trochanteric region.
    outline = ndimage.binary_fill_holes(
        ndimage.binary_opening(smooth > background + 0.3 * (shaft_level - background), iterations=2)
    )

    def axis(y: float) -> float:
        return float(np.polyval(coef, y))

    # Greater trochanter: follow the lateral outline of the femur up from the
    # shaft; the outline ends at the trochanter tip, above which there is soft
    # tissue on the lateral side.
    trochanter = None
    y = int(rows[0])
    edge = None
    for x in range(int(np.clip(axis(y), 0, width - 1)), -1, -1):
        if not outline[y, x]:
            edge = x + 1
            break
    if edge is not None:
        top = y
        while y > 2:
            y -= 1
            low = max(0, edge - 10)
            high = int(np.clip(min(edge + 14, axis(y)), low + 1, width))
            window = np.where(outline[y, low:high])[0]
            if window.size == 0:
                break
            edge = low + int(window[0])
            top = y
        trochanter = (float(edge), float(top))
    grad_y, grad_x = np.gradient(ndimage.gaussian_filter(image, 2.0))
    magnitude = np.hypot(grad_x, grad_y)
    edge_y, edge_x = np.where(magnitude > np.percentile(magnitude, 85))
    reference = trochanter[1] if trochanter else height * 0.3
    inside = ndimage.uniform_filter(image, size=15)
    grid_y, grid_x = np.mgrid[0:height, 0:width]
    outside = (
        (grid_x < np.polyval(coef, grid_y) + 10)
        | (grid_x > np.polyval(coef, grid_y) + 65)
        | (grid_y < reference - 35)
        | (grid_y > reference + 25)
    )
    if trochanter is not None:
        expected_x, expected_y = trochanter[0] + 55.0, trochanter[1] - 5.0
        prior = np.exp(-((grid_x - expected_x) ** 2 + (grid_y - expected_y) ** 2) / (2 * 18.0**2))
    else:
        prior = np.ones((height, width))
    best: tuple[float, tuple[float, float, float] | None] = (0.0, None)
    unit_x = grad_x[edge_y, edge_x] / (magnitude[edge_y, edge_x] + 1e-9)
    unit_y = grad_y[edge_y, edge_x] / (magnitude[edge_y, edge_x] + 1e-9)
    for radius in range(19, 30):
        votes = np.zeros((height, width))
        for sign in (1, -1):
            cx = np.round(edge_x - sign * radius * unit_x).astype(int)
            cy = np.round(edge_y - sign * radius * unit_y).astype(int)
            ok = (cx >= 0) & (cx < width) & (cy >= 0) & (cy < height)
            np.add.at(votes, (cy[ok], cx[ok]), 1)
        votes = ndimage.gaussian_filter(votes, 1.2)
        votes[outside] = 0
        # Anatomical prior: the head centre lies about 55 mm medial to the
        # trochanter tip and close to its height.
        votes = votes * prior
        peak = np.unravel_index(int(np.argmax(votes)), votes.shape)
        score = float(votes[peak]) / radius * (0.5 + float(inside[peak]))
        if score > best[0]:
            best = (score, (float(peak[1]), float(peak[0]), float(radius)))
    return {
        "coef": coef,
        "shaft_width": shaft_width,
        "trochanter": trochanter,
        "head": best[1],
        "bone": bone,
        "shaft_level": shaft_level,
        "outline": outline,
    }


def _lesser_trochanter_bottom(landmarks: dict[str, Any], head: tuple[float, float, float]) -> int | None:
    bone = landmarks["bone"]
    height, width = bone.shape
    coef = landmarks["coef"]
    start = int(head[1] + head[2])
    extents = []
    for y in range(start, min(height - 1, start + 110)):
        xa = int(np.clip(np.polyval(coef, y), 0, width - 1))
        x = xa
        while x < width - 1 and bone[y, x + 1] and x - xa < 70:
            x += 1
        extents.append((y, x - xa))
    if len(extents) < 6:
        return None
    ys = np.asarray([item[0] for item in extents])
    reach = ndimage.median_filter(np.asarray([item[1] for item in extents], dtype=float), 5)
    peak = int(np.argmax(reach[:60]))
    back = np.where(reach[peak:] <= landmarks["shaft_width"] / 2 + 5)[0]
    return int(ys[peak + back[0]]) if back.size else None


def hip_markup(
    normalized: np.ndarray,
    region: str,
    spacing_y_mm: float = DEFAULT_PIXEL_SPACING_Y_MM,
    spacing_x_mm: float = DEFAULT_PIXEL_SPACING_X_MM,
) -> Markup:
    grid = _Grid(normalized.shape, spacing_y_mm, spacing_x_mm, mirror=region == LEFT_HIP)
    image = grid.resample(normalized)
    height, width = image.shape
    landmarks = _hip_landmarks(image)
    head = landmarks["head"]
    trochanter = landmarks["trochanter"]
    markup = Markup(region=region)
    if head is None or trochanter is None:
        markup.notes.append("Головка или большой вертел бедра не найдены автоматически.")
        return finalize(markup, normalized.shape, spacing_y_mm, spacing_x_mm)

    hx, hy, radius = head
    tx, ty = trochanter
    lesser = _lesser_trochanter_bottom(landmarks, head)
    bottom = min(height - 1.0, max((lesser + 10) if lesser else 0.0, ty + 65))
    top = max(0.0, hy - radius - 5)
    lateral = max(0.0, tx - 5)
    medial = min(width - 1.0, hx + radius + 5)
    markup.shapes.append(
        {
            "label": HIP_ROI_LABEL,
            "type": "POLYGON",
            "style": STYLE_ROI,
            "points": [
                grid.to_pixels(lateral, top),
                grid.to_pixels(medial, top),
                grid.to_pixels(medial, bottom),
                grid.to_pixels(lateral, bottom),
            ],
        }
    )
    # Neck box: 15 mm along the neck axis, just lateral to the head.
    py = ty + 25
    px = float(np.polyval(landmarks["coef"], py))
    along = np.asarray([px - hx, py - hy], dtype=float)
    along /= float(np.linalg.norm(along)) + 1e-9
    across = np.asarray([-along[1], along[0]])
    centre = np.asarray([hx, hy]) + along * (radius + 10)
    corners = [
        centre + along * 7.5 + across * 18,
        centre + along * 7.5 - across * 18,
        centre - along * 7.5 - across * 18,
        centre - along * 7.5 + across * 18,
    ]
    markup.shapes.append(
        {
            "label": NECK_LABEL,
            "type": "POLYGON",
            "style": STYLE_ROI,
            "points": [grid.to_pixels(float(c[0]), float(c[1])) for c in corners],
        }
    )
    # Ward's area: the least dense 10 mm square in the neck just inferior to
    # the neck box, as the densitometer places it.
    bone = landmarks["bone"]
    smooth = ndimage.uniform_filter(image, size=10)
    best_ward: tuple[float, np.ndarray] | None = None
    for shift_across in np.arange(4.0, 22.0, 2.0):
        for shift_along in np.arange(-10.0, 6.0, 2.0):
            point = centre + across * shift_across * np.sign(across[1] or 1.0) + along * shift_along
            x, y = int(round(point[0])), int(round(point[1]))
            if not (5 <= x < width - 5 and 5 <= y < height - 5) or not bone[y - 5 : y + 5, x - 5 : x + 5].all():
                continue
            value = float(smooth[y, x])
            if best_ward is None or value < best_ward[0]:
                best_ward = (value, point)
    if best_ward is not None:
        wx, wy = best_ward[1]
        markup.derived.append(
            {
                "label": WARD_LABEL,
                "type": "POLYGON",
                "style": STYLE_ROI,
                "points": [
                    grid.to_pixels(wx - 5, wy - 5),
                    grid.to_pixels(wx + 5, wy - 5),
                    grid.to_pixels(wx + 5, wy + 5),
                    grid.to_pixels(wx - 5, wy + 5),
                ],
            }
        )
    # Bone edge of the proximal femur and the pelvis inside the analysed field.
    contour = _trace_boundary(landmarks["outline"])
    segment: list[tuple[float, float]] = []
    for x, y in contour + [(-1.0, -1.0)]:
        # The field border is not a bone edge: split the outline there.
        if 1.5 < x < width - 2.5 and 1.5 < y < height - 2.5:
            segment.append((x, y))
            continue
        if len(segment) >= 4:
            markup.derived.append(
                {
                    "label": BONE_EDGE_LABEL,
                    "type": "POLYLINE",
                    "style": STYLE_EDGE,
                    "points": [grid.to_pixels(px, py) for px, py in segment],
                }
            )
        segment = []
    markup.derived.append(
        {
            "label": HEAD_LABEL,
            "type": "ELLIPSE",
            "style": STYLE_LANDMARK,
            "points": [
                grid.to_pixels(hx - radius, hy),
                grid.to_pixels(hx + radius, hy),
                grid.to_pixels(hx, hy - radius),
                grid.to_pixels(hx, hy + radius),
            ],
        }
    )
    markup.metrics = {
        "found": 1.0,
        "above_mm": float(top),
        "below_mm": float(height - 1 - bottom),
        "lateral_mm": float(lateral),
        "head_radius_mm": float(radius),
        "lesser_found": float(lesser is not None),
        "trochanter_to_head_mm": float(hx - tx),
    }
    markup.confidence = "high" if lesser is not None else "low"
    if lesser is None:
        markup.notes.append("Малый вертел не выделен; нижняя граница области интереса оценена.")
    return finalize(markup, normalized.shape, spacing_y_mm, spacing_x_mm)


# ------------------------------------------------------- checks / editing ----


def _check(criterion: str, measured: float | None, required: float, unit: str = "мм") -> dict[str, Any]:
    ok = None if measured is None else bool(measured >= required)
    return {
        "criterion": criterion,
        "measured": None if measured is None else round(float(measured), 1),
        "required": required,
        "unit": unit,
        "ok": ok,
    }


def finalize(
    markup: Markup,
    shape: tuple[int, int],
    spacing_y_mm: float = DEFAULT_PIXEL_SPACING_Y_MM,
    spacing_x_mm: float = DEFAULT_PIXEL_SPACING_X_MM,
) -> Markup:
    """Rebuild derived shapes and re-evaluate the ТЗ checks from the editable shapes.

    Called after automatic markup and again after a specialist has moved points,
    so the checks always describe the markup that is actually shown.
    """
    grid = _Grid(shape, spacing_y_mm, spacing_x_mm, mirror=markup.region == LEFT_HIP)
    by_label = {item["label"]: item for item in markup.shapes}
    markup.checks = []
    if markup.region == SPINE:
        markup.derived = [item for item in markup.derived if item["label"] not in VERTEBRA_LABELS]
        discs = [by_label[label] for label in SPINE_DISC_LABELS if label in by_label]
        levels = []
        for disc in discs:
            (x0, y0), (x1, y1) = (grid.to_mm(*disc["points"][0]), grid.to_mm(*disc["points"][-1]))
            levels.append((min(x0, x1), max(x0, x1), (y0 + y1) / 2, disc))
        levels.sort(key=lambda item: item[2])
        for index in range(len(levels) - 1):
            upper, lower = levels[index][3], levels[index + 1][3]
            if index < len(VERTEBRA_LABELS) and len(levels) == len(SPINE_DISC_LABELS):
                markup.derived.append(
                    {
                        "label": VERTEBRA_LABELS[index],
                        "type": "POLYGON",
                        "style": STYLE_ROI,
                        "points": [upper["points"][0], upper["points"][-1], lower["points"][-1], lower["points"][0]],
                    }
                )
        gaps = [levels[i + 1][2] - levels[i][2] for i in range(len(levels) - 1)]
        vertebra = float(np.mean(gaps)) if gaps else None
        top_level = levels[0][2] if levels else None
        markup.checks.append(
            _check(
                "Над L1 видна половина тела Th12",
                top_level,
                round(0.5 * vertebra, 1) if vertebra else 15.0,
            )
        )
        crest = by_label.get(CREST_LABEL)
        markup.checks.append(
            {
                "criterion": "В поле видны верхние края подвздошных костей",
                "measured": None,
                "required": None,
                "unit": "",
                "ok": crest is not None,
            }
        )
    else:
        markup.derived = [item for item in markup.derived if item["label"] != TROCHANTER_LINE_LABEL]
        roi = by_label.get(HIP_ROI_LABEL)
        if roi:
            # The trochanteric line runs across the analysed box from its upper
            # lateral corner to its lower medial corner, as on the densitometer.
            markup.derived.append(
                {
                    "label": TROCHANTER_LINE_LABEL,
                    "type": "POLYLINE",
                    "style": STYLE_ROI,
                    "points": [list(roi["points"][0]), list(roi["points"][2])],
                }
            )
            points = [grid.to_mm(*point) for point in roi["points"]]
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            markup.checks.extend(
                [
                    _check("Поле выше области интереса", min(ys), HIP_SUPERIOR_MARGIN_CM * 10),
                    _check(
                        "Поле ниже области интереса",
                        grid.height_mm - 1 - max(ys),
                        HIP_INFERIOR_MARGIN_CM * 10,
                    ),
                    _check("Поле от бокового края", min(xs), HIP_LATERAL_MARGIN_CM * 10),
                ]
            )
    return markup


def build_markup(
    normalized: np.ndarray,
    region: str,
    spacing_y_mm: float = DEFAULT_PIXEL_SPACING_Y_MM,
    spacing_x_mm: float = DEFAULT_PIXEL_SPACING_X_MM,
) -> Markup:
    """Automatic markup; never raises, a failure becomes an empty low-confidence markup."""
    try:
        if region == SPINE:
            return spine_markup(normalized, spacing_y_mm, spacing_x_mm)
        return hip_markup(normalized, region, spacing_y_mm, spacing_x_mm)
    except Exception as exc:  # the report must still be produced
        markup = Markup(region=region)
        markup.notes.append(f"Автоматическая разметка не построена: {exc}")
        return markup


def edited_markup(
    original: dict[str, Any],
    shapes: list[dict[str, Any]],
    shape: tuple[int, int],
    spacing_y_mm: float = DEFAULT_PIXEL_SPACING_Y_MM,
    spacing_x_mm: float = DEFAULT_PIXEL_SPACING_X_MM,
) -> Markup:
    """Apply a specialist's edit: same shape labels, moved points, checks recomputed."""
    rows, columns = shape
    known = {item["label"]: item for item in original.get("shapes", [])}
    cleaned = []
    for item in shapes:
        label = str(item.get("label", ""))
        if label not in known:
            raise ValueError(f"Неизвестный элемент разметки: {label}")
        points = item.get("points") or []
        if len(points) != len(known[label]["points"]):
            raise ValueError(f"Число точек элемента «{label}» изменилось")
        clipped = [
            [round(float(np.clip(float(x), 0, columns - 1)), 2), round(float(np.clip(float(y), 0, rows - 1)), 2)]
            for x, y in points
        ]
        cleaned.append(
            {"label": label, "type": known[label]["type"], "style": known[label].get("style", STYLE_ROI), "points": clipped}
        )
    derived = [dict(item) for item in original.get("derived", []) if item["label"] in STATIC_LABELS]
    # Ward's square belongs to the neck: it follows the neck box when moved.
    old_neck = known.get(NECK_LABEL)
    new_neck = next((item for item in cleaned if item["label"] == NECK_LABEL), None)
    for item in original.get("derived", []):
        if item["label"] == WARD_LABEL and old_neck and new_neck:
            dx = float(np.mean([p[0] for p in new_neck["points"]]) - np.mean([p[0] for p in old_neck["points"]]))
            dy = float(np.mean([p[1] for p in new_neck["points"]]) - np.mean([p[1] for p in old_neck["points"]]))
            derived.append({**item, "points": [[round(x + dx, 2), round(y + dy, 2)] for x, y in item["points"]]})
    markup = Markup(
        region=original.get("region", ""),
        shapes=cleaned,
        derived=derived,
        confidence=original.get("confidence", "low"),
        notes=list(original.get("notes", [])),
        source="specialist",
    )
    return finalize(markup, shape, spacing_y_mm, spacing_x_mm)
