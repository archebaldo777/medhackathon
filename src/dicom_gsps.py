"""Markup as a DICOM Grayscale Softcopy Presentation State (ТЗ 2.6).

The densitometer's markup is an annotation layer over the image, not part of the
pixels. The standard DICOM object for such a layer is a Grayscale Softcopy
Presentation State: it references the source image and carries polylines,
ellipses and text in image pixel coordinates, so a PACS viewer draws the markup
over the original study and a specialist can switch it on and off. The service
writes the automatic markup this way and rewrites it after the specialist has
confirmed or edited it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from pydicom import dcmread
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

from .dicom_sr import safe_date, safe_time, safe_uid

GRAYSCALE_SOFTCOPY_PRESENTATION_STATE = "1.2.840.10008.5.1.4.1.1.11.1"
MARKUP_SERIES_NUMBER = 902
MARKUP_LAYER = "EVECTIO_MARKUP"


def _graphic(shape: dict[str, Any]) -> Dataset:
    # DICOM places the centre of the first pixel at (0.5, 0.5); markup points
    # are 0-based pixel centres, hence the half-pixel shift.
    points = [[float(x) + 0.5, float(y) + 0.5] for x, y in shape["points"]]
    kind = shape.get("type", "POLYLINE")
    if kind == "POLYGON":
        points = points + [points[0]]
        kind = "POLYLINE"
    item = Dataset()
    item.GraphicAnnotationUnits = "PIXEL"
    item.GraphicDimensions = 2
    item.NumberOfGraphicPoints = len(points)
    item.GraphicData = [value for point in points for value in point]
    item.GraphicType = kind
    item.GraphicFilled = "N"
    return item


def _text(shape: dict[str, Any]) -> Dataset:
    xs = [float(point[0]) for point in shape["points"]]
    ys = [float(point[1]) for point in shape["points"]]
    item = Dataset()
    item.UnformattedTextValue = str(shape["label"])[:64]
    item.AnchorPointAnnotationUnits = "PIXEL"
    item.AnchorPoint = [max(xs) + 2.5, (min(ys) + max(ys)) / 2 + 0.5]
    item.AnchorPointVisibility = "N"
    return item


def save_markup_gsps(
    source_path: str | Path,
    markup: dict[str, Any],
    output_path: str | Path,
    creator: str = "Evectio",
    series_uid: str | None = None,
    instance_number: int = 1,
) -> Path:
    """Write ``markup`` as a GSPS that references the source image."""
    source = dcmread(source_path, stop_before_pixels=True, force=True)
    rows, columns = markup.get("image_shape") or [int(source.Rows), int(source.Columns)]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    sop_uid = generate_uid()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = GRAYSCALE_SOFTCOPY_PRESENTATION_STATE
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.ImplementationClassUID = generate_uid()

    state = FileDataset(str(output), {}, file_meta=meta, preamble=b"\0" * 128)
    state.SpecificCharacterSet = "ISO_IR 192"
    state.SOPClassUID = GRAYSCALE_SOFTCOPY_PRESENTATION_STATE
    state.SOPInstanceUID = sop_uid
    state.StudyDate = safe_date(getattr(source, "StudyDate", ""))
    state.StudyTime = safe_time(getattr(source, "StudyTime", ""))
    state.AccessionNumber = ""
    state.Modality = "PR"
    state.Manufacturer = "Evectio"
    state.ReferringPhysicianName = ""
    state.SeriesDescription = "Evectio markup"
    state.PatientName = ""
    state.PatientID = ""
    state.PatientBirthDate = ""
    state.PatientSex = ""
    state.StudyInstanceUID = safe_uid(getattr(source, "StudyInstanceUID", ""))
    state.SeriesInstanceUID = series_uid or generate_uid()
    state.StudyID = ""
    state.SeriesNumber = MARKUP_SERIES_NUMBER
    state.InstanceNumber = int(instance_number)

    confirmed = markup.get("status") in {"accepted", "edited"}
    state.ContentLabel = "EVECTIO_MARKUP"
    state.ContentDescription = (
        "Разметка подтверждена специалистом"
        if markup.get("status") == "accepted"
        else "Разметка исправлена специалистом"
        if markup.get("status") == "edited"
        else "Автоматическая разметка, требует подтверждения"
    )[:64]
    state.ContentCreatorName = (creator if confirmed else "Evectio")[:64]
    now = datetime.now().astimezone()
    state.PresentationCreationDate = now.strftime("%Y%m%d")
    state.PresentationCreationTime = now.strftime("%H%M%S")

    referenced_image = Dataset()
    referenced_image.ReferencedSOPClassUID = safe_uid(getattr(source, "SOPClassUID", ""))
    referenced_image.ReferencedSOPInstanceUID = safe_uid(getattr(source, "SOPInstanceUID", ""))
    referenced_series = Dataset()
    referenced_series.SeriesInstanceUID = safe_uid(getattr(source, "SeriesInstanceUID", ""))
    referenced_series.ReferencedImageSequence = [referenced_image]
    state.ReferencedSeriesSequence = [referenced_series]

    area = Dataset()
    area.DisplayedAreaTopLeftHandCorner = [1, 1]
    area.DisplayedAreaBottomRightHandCorner = [int(columns), int(rows)]
    area.PresentationSizeMode = "SCALE TO FIT"
    spacing_y = float(markup.get("pixel_spacing_y_mm") or 1.05)
    spacing_x = float(markup.get("pixel_spacing_x_mm") or 0.6)
    area.PresentationPixelSpacing = [spacing_y, spacing_x]
    state.DisplayedAreaSelectionSequence = [area]

    layer = Dataset()
    layer.GraphicLayer = MARKUP_LAYER
    layer.GraphicLayerOrder = 1
    layer.GraphicLayerDescription = "Разметка областей измерения DXA"
    state.GraphicLayerSequence = [layer]

    shapes = list(markup.get("shapes", [])) + list(markup.get("derived", []))
    annotation = Dataset()
    annotation.GraphicLayer = MARKUP_LAYER
    annotation.GraphicObjectSequence = [_graphic(shape) for shape in shapes if shape.get("points")]
    annotation.TextObjectSequence = [_text(shape) for shape in shapes if shape.get("points")]
    state.GraphicAnnotationSequence = [annotation]
    state.PresentationLUTShape = "IDENTITY"
    state.save_as(output, enforce_file_format=True)
    return output
