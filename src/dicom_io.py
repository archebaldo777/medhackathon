from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pydicom
from pydicom.errors import InvalidDicomError

from .constants import (
    DEFAULT_PIXEL_SPACING_X_MM,
    DEFAULT_PIXEL_SPACING_Y_MM,
    LEFT_HIP,
    RIGHT_HIP,
    SPINE,
)


class DicomReadError(RuntimeError):
    """A controlled error raised when a DICOM image cannot be decoded."""


@dataclass
class DicomImage:
    path: Path
    pixels: np.ndarray
    study_uid: str
    image_uid: str
    rows: int
    columns: int
    photometric: str
    pixel_spacing_y_mm: float
    pixel_spacing_x_mm: float
    pixel_spacing_source: str
    # Geometry/description tags used to report projection and laterality.
    tags: dict[str, Any] = field(default_factory=dict)


# Tags read for projection and laterality; none of them identifies the patient.
PROJECTION_TAGS = (
    "ViewPosition",
    "PatientOrientation",
    "ImageLaterality",
    "Laterality",
    "SeriesDescription",
    "ProtocolName",
    "StudyDescription",
    "AcquisitionDeviceProcessingDescription",
)

DICOM_SUFFIXES = {".dcm", ".dicom", ".dic"}


def is_dicom_file(path: str | Path) -> bool:
    """True for a DICOM file: a known suffix or the Part 10 ``DICM`` signature.

    Scanner exports often have no extension at all (``IM000001``), so the
    signature at byte 128 is checked for any other file. DICOMDIR is an index,
    not an image, and is skipped.
    """
    source = Path(path)
    if not source.is_file() or source.name.upper() == "DICOMDIR" or source.name.startswith("."):
        return False
    if source.suffix.lower() in DICOM_SUFFIXES:
        return True
    try:
        with source.open("rb") as handle:
            handle.seek(128)
            if handle.read(4) == b"DICM":
                return True
    except OSError:
        return False
    if source.suffix:
        return False
    # Legacy files without the Part 10 preamble and without an extension: accept
    # them when the header parses and names an image object.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            header = pydicom.dcmread(source, force=True, stop_before_pixels=True)
        return bool(getattr(header, "SOPClassUID", None)) and "Rows" in header
    except Exception:
        return False


def _safe_text(value: object, fallback: str = "") -> str:
    text = str(value).strip() if value is not None else ""
    return text or fallback


def _pixel_spacing(dataset: pydicom.dataset.Dataset) -> tuple[float, float, str]:
    """Return calibrated (Y, X) pixel size in millimetres.

    Standard DICOM spacing fields take precedence when valid. Challenge files
    omit them, in which case the fixed calibration supplied for the scanner is
    used. DICOM stores spacing in row (Y), column (X) order.
    """
    for keyword in ("PixelSpacing", "ImagerPixelSpacing", "NominalScannedPixelSpacing"):
        value = getattr(dataset, keyword, None)
        try:
            if value is not None and len(value) >= 2:
                y_mm, x_mm = float(value[0]), float(value[1])
                if np.isfinite(y_mm) and np.isfinite(x_mm) and y_mm > 0 and x_mm > 0:
                    return y_mm, x_mm, keyword
        except (TypeError, ValueError):
            continue
    return DEFAULT_PIXEL_SPACING_Y_MM, DEFAULT_PIXEL_SPACING_X_MM, "scanner_default"


def distance_cm_to_pixels(distance_cm: float, spacing_mm: float) -> float:
    """Convert a physical distance to pixels for one calibrated image axis."""
    if distance_cm < 0:
        raise ValueError("Расстояние не может быть отрицательным")
    if not np.isfinite(spacing_mm) or spacing_mm <= 0:
        raise ValueError("Размер пикселя должен быть положительным")
    return distance_cm * 10.0 / spacing_mm


def read_dicom(path: str | Path) -> DicomImage:
    source = Path(path)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                dataset = pydicom.dcmread(source, force=False)
            except InvalidDicomError:
                # Some valid legacy datasets omit the Part 10 preamble.
                dataset = pydicom.dcmread(source, force=True)
            pixels = np.asarray(dataset.pixel_array, dtype=np.float32)
            # Empty when the tag is missing: the report carries only real DICOM identifiers.
            study_uid = _safe_text(getattr(dataset, "StudyInstanceUID", None))
            image_uid = _safe_text(getattr(dataset, "SOPInstanceUID", None))
            samples_per_pixel = int(getattr(dataset, "SamplesPerPixel", 1) or 1)
            frames = int(getattr(dataset, "NumberOfFrames", 1) or 1)
            photometric = _safe_text(getattr(dataset, "PhotometricInterpretation", "MONOCHROME2"))
            spacing_y_mm, spacing_x_mm, spacing_source = _pixel_spacing(dataset)
            tags = {keyword: getattr(dataset, keyword, None) for keyword in PROJECTION_TAGS}
    except (InvalidDicomError, AttributeError, ValueError, OSError, RuntimeError) as exc:
        raise DicomReadError(f"Не удалось прочитать DICOM: {exc}") from exc

    # Multi-frame: the first frame. pixel_array is (frames, rows, columns[, samples]).
    if frames > 1 and pixels.ndim >= 3:
        pixels = pixels[0]
    # Colour (RGB, or YBR already converted to RGB by pydicom): luminance, ITU-R BT.601.
    if samples_per_pixel > 1 and pixels.ndim == 3 and pixels.shape[-1] in (3, 4):
        pixels = pixels[..., :3] @ np.asarray([0.299, 0.587, 0.114], dtype=np.float32)
        photometric = "MONOCHROME2"
    if pixels.ndim != 2 or pixels.size == 0:
        raise DicomReadError(f"Ожидалось монохромное 2D-изображение, получена форма {pixels.shape}")
    if not np.isfinite(pixels).all():
        pixels = np.nan_to_num(pixels, copy=False)

    slope = float(getattr(dataset, "RescaleSlope", 1.0) or 1.0)
    intercept = float(getattr(dataset, "RescaleIntercept", 0.0) or 0.0)
    pixels = pixels * slope + intercept
    if photometric.upper() == "MONOCHROME1":
        pixels = float(pixels.max()) + float(pixels.min()) - pixels

    return DicomImage(
        path=source,
        pixels=pixels,
        study_uid=study_uid,
        image_uid=image_uid,
        rows=int(pixels.shape[0]),
        columns=int(pixels.shape[1]),
        photometric=photometric,
        pixel_spacing_y_mm=spacing_y_mm,
        pixel_spacing_x_mm=spacing_x_mm,
        pixel_spacing_source=spacing_source,
        tags=tags,
    )


def normalize_pixels(pixels: np.ndarray) -> np.ndarray:
    image = np.asarray(pixels, dtype=np.float32)
    low, high = np.percentile(image, (1.0, 99.0))
    if high <= low:
        low, high = float(image.min()), float(image.max())
    if high <= low:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - low) / (high - low), 0.0, 1.0).astype(np.float32)


def anatomical_region(normalized: np.ndarray) -> str:
    """Detect lumbar spine vs hip and, for hip, infer laterality.

    DXA spine acquisitions are vertically centred, while hip acquisitions carry a
    strong femoral shaft on one side in the lower half. The column-count hint is
    only used near the ambiguous centroid band and is not based on patient data.
    """
    height, width = normalized.shape
    lower = normalized[int(height * 0.55) :, :]
    threshold = float(np.quantile(lower, 0.75))
    weights = np.clip(lower - threshold, 0.0, None)
    mass_x = weights.sum(axis=0)
    total = float(mass_x.sum())
    centroid = 0.5 if total <= 1e-8 else float(np.dot(np.arange(width), mass_x) / (total * max(width - 1, 1)))

    # The supplied scanner produces a very stable 300 px spine field and 280 px
    # hip field. Centroid keeps the rule usable when dimensions are rescaled.
    if width >= 290:
        return SPINE
    if width / max(height, 1) < 0.83 and 0.43 <= centroid <= 0.57:
        return SPINE
    return RIGHT_HIP if centroid < 0.5 else LEFT_HIP
