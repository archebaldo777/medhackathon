"""Projection (view) of a DXA image and laterality of a hip scan (ТЗ 2.2).

DXA densitometry of the lumbar spine and the proximal femur is acquired in the
frontal (posterior-anterior beam, reported as "AP") projection; the lumbar spine
may additionally be scanned laterally. The projection is taken from DICOM
geometry tags, which every organiser file carries:

1. ``ViewPosition`` (0018,5101) — AP / PA / LL / RL / LATERAL;
2. ``PatientOrientation`` (0020,0020) — the patient directions of the image rows
   and columns. A frontal image lies in the coronal plane (rows L/R, columns
   H/F); a lateral image lies in the sagittal plane (rows A/P, columns H/F);
3. textual descriptions (series, protocol, study) with AP/PA/LAT keywords.

Without any of these tags the standard DXA protocol projection is reported and
the source is marked as ``protocol`` so a reader knows it was not measured.
The quality model was trained on frontal images only, so a lateral image is
analysed but flagged for mandatory manual review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .constants import LEFT_HIP, RIGHT_HIP

FRONTAL = "AP"
LATERAL = "LAT"
UNKNOWN = "UNKNOWN"

PROJECTION_NAMES_RU = {
    FRONTAL: "Прямая (AP)",
    LATERAL: "Боковая (LAT)",
    UNKNOWN: "Не определена",
}

_VIEW_POSITION = {
    "AP": FRONTAL,
    "PA": FRONTAL,
    "LL": LATERAL,
    "RL": LATERAL,
    "LAT": LATERAL,
    "LATERAL": LATERAL,
    "RLD": LATERAL,
    "LLD": LATERAL,
}

_LATERAL_WORDS = re.compile(r"(?<![A-ZА-Я])(LAT|LATERAL|LVA|БОК\w*)(?![A-ZА-Я])", re.IGNORECASE)
_FRONTAL_WORDS = re.compile(r"(?<![A-ZА-Я])(AP|PA|ПРЯМ\w*)(?![A-ZА-Я])", re.IGNORECASE)


@dataclass(frozen=True)
class ProjectionInfo:
    code: str
    name_ru: str
    source: str

    def as_dict(self) -> dict[str, str]:
        return {"code": self.code, "name": self.name_ru, "source": self.source}


def _info(code: str, source: str) -> ProjectionInfo:
    return ProjectionInfo(code, PROJECTION_NAMES_RU[code], source)


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)) or type(value).__name__ == "MultiValue":
        return "\\".join(str(item) for item in value)
    return str(value).strip()


def _from_orientation(value: Any) -> str | None:
    """Map PatientOrientation (row direction, column direction) to a projection."""
    text = _text(value).upper()
    parts = [part for part in re.split(r"[\\\s,]+", text) if part]
    if len(parts) != 2:
        return None
    row, column = parts[0][:1], parts[1][:1]
    axes = {row, column}
    if not axes <= set("LRAPHF") or len(axes) != 2:
        return None
    if axes & {"L", "R"} and axes & {"H", "F"}:
        return FRONTAL
    if axes & {"A", "P"} and axes & {"H", "F"}:
        return LATERAL
    return None


def projection_from_tags(tags: dict[str, Any]) -> ProjectionInfo | None:
    view = _text(tags.get("ViewPosition")).upper()
    if view in _VIEW_POSITION:
        return _info(_VIEW_POSITION[view], "dicom:ViewPosition")
    orientation = _from_orientation(tags.get("PatientOrientation"))
    if orientation:
        return _info(orientation, "dicom:PatientOrientation")
    for keyword in ("SeriesDescription", "ProtocolName", "StudyDescription", "AcquisitionDeviceProcessingDescription"):
        text = _text(tags.get(keyword))
        if not text:
            continue
        if _LATERAL_WORDS.search(text):
            return _info(LATERAL, f"dicom:{keyword}")
        if _FRONTAL_WORDS.search(text):
            return _info(FRONTAL, f"dicom:{keyword}")
    return None


def detect_projection(tags: dict[str, Any]) -> ProjectionInfo:
    """Projection from DICOM tags, otherwise the standard DXA protocol projection."""
    return projection_from_tags(tags) or _info(FRONTAL, "protocol")


def laterality_from_tags(tags: dict[str, Any]) -> str | None:
    """Hip side from (Image)Laterality, if the scanner filled it in."""
    for keyword in ("ImageLaterality", "Laterality"):
        value = _text(tags.get(keyword)).upper()
        if value == "R":
            return RIGHT_HIP
        if value == "L":
            return LEFT_HIP
    return None
