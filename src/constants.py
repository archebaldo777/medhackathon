from __future__ import annotations

from dataclasses import dataclass


SPINE = "lumbar_spine"
RIGHT_HIP = "right_proximal_femur"
LEFT_HIP = "left_proximal_femur"

# The challenge data come from one scanner. Its DICOM exports do not contain
# PixelSpacing, so these values are the authoritative fallback (row/Y first,
# column/X second, as in DICOM).
DEFAULT_PIXEL_SPACING_Y_MM = 1.05
DEFAULT_PIXEL_SPACING_X_MM = 0.6
# ТЗ 2.3, рис. 6: 3 cm above and below the region of interest, 2 cm from the edge.
HIP_SUPERIOR_MARGIN_CM = 3.0
HIP_INFERIOR_MARGIN_CM = 3.0
HIP_LATERAL_MARGIN_CM = 2.0

# ТЗ 2.3: the lumbar axis is accepted up to a 5 degree deviation from vertical.
AXIS_TOLERANCE_DEG = 5.0

SPINE_NAME_RU = "Поясничный отдел позвоночника"
HIP_NAME_RU = "Проксимальный отдел бедра"
# The report names the hip side: both hips of one study are separate rows.
ANATOMICAL_REGION_NAMES_RU = {
    SPINE: SPINE_NAME_RU,
    RIGHT_HIP: f"{HIP_NAME_RU} (правый)",
    LEFT_HIP: f"{HIP_NAME_RU} (левый)",
}

SPINE_LABELS = (
    "spine_positioning",
    "spine_axis_deviation",
    "foreign_object_or_artifact",
)
HIP_LABELS = (
    "hip_positioning_or_rotation",
    "hip_roi_incorrect",
)


@dataclass(frozen=True)
class ViolationInfo:
    code: str
    title_ru: str
    explanation_ru: str


VIOLATIONS = {
    "spine_positioning": ViolationInfo(
        "SPINE_POSITIONING",
        "Некорректная укладка",
        "Границы сканирования могут не включать верхние края подвздошных костей и/или половину Th12.",
    ),
    "spine_axis_deviation": ViolationInfo(
        "SPINE_AXIS_DEVIATION",
        "Не выравнена ось позвоночника",
        "Ось поясничного отдела, вероятно, отклонена от вертикали более допустимого значения.",
    ),
    "foreign_object_or_artifact": ViolationInfo(
        "FOREIGN_OBJECT_OR_ARTIFACT",
        "Присутствуют посторонние предметы",
        "Обнаружены признаки постороннего предмета, выраженного артефакта или наложения.",
    ),
    "hip_positioning_or_rotation": ViolationInfo(
        "HIP_POSITIONING_OR_ROTATION",
        "Некорректная укладка",
        "Взаимное положение шейки и вертелов может не соответствовать стандартной укладке.",
    ),
    "hip_roi_incorrect": ViolationInfo(
        "HIP_ROI_INCORRECT",
        "Некорректная область интереса",
        "Запас изображения вокруг предполагаемой области интереса может быть недостаточным.",
    ),
}

REPORT_COLUMNS = (
    "path_to_study",
    "study_uid",
    "image_uid",
    "anatomical_region",
    "quality_class",
    "violation_type",
    "processing_status",
    "time_of_processing",
    "quality_prob",
    "projection",
)
