"""DICOM copy of the visual explanation (ТЗ 2.6, «визуализация результата в DICOM»).

The PNG in ``visualizations/`` is convenient in a browser but a PACS viewer
cannot show it next to the study. This module wraps the same picture into a
Secondary Capture image that belongs to the source study (same
StudyInstanceUID) and to a new derived series, so the specialist opens the
original image and the model's explanation side by side in the usual viewer.
Patient identifiers are not copied: the service is a local research prototype
and its outputs are de-identified like the SR report.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
from pydicom import dcmread
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from .analyzer import AnalysisResult
from .dicom_sr import safe_date, safe_time, safe_uid

VISUALIZATION_SERIES_NUMBER = 901


def save_dicom_visualization(
    source_path: str | Path,
    result: AnalysisResult,
    png_path: str | Path,
    output_path: str | Path,
    series_uid: str | None = None,
    instance_number: int = 1,
) -> Path:
    """Write the PNG explanation as an RGB Secondary Capture in the source study."""
    source = dcmread(source_path, stop_before_pixels=True, force=True)
    pixels = np.asarray(Image.open(png_path).convert("RGB"), dtype=np.uint8)
    rows, columns = int(pixels.shape[0]), int(pixels.shape[1])
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    sop_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    image = FileDataset(str(output), {}, file_meta=file_meta, preamble=b"\0" * 128)
    image.SpecificCharacterSet = "ISO_IR 192"
    image.ImageType = ["DERIVED", "SECONDARY"]
    image.SOPClassUID = SecondaryCaptureImageStorage
    image.SOPInstanceUID = sop_uid
    image.StudyDate = safe_date(getattr(source, "StudyDate", ""))
    image.StudyTime = safe_time(getattr(source, "StudyTime", ""))
    now = datetime.now().astimezone()
    image.ContentDate = now.strftime("%Y%m%d")
    image.ContentTime = now.strftime("%H%M%S.%f")
    image.AccessionNumber = ""
    image.Modality = "OT"
    image.ConversionType = "WSD"
    image.Manufacturer = "Evectio"
    image.ReferringPhysicianName = ""
    image.SeriesDescription = "AI DXA quality control - visual explanation"
    image.DerivationDescription = (
        f"Evectio {result.model_version}: quality_class={result.quality_class}, "
        f"quality_prob={float(result.quality_prob or 0.0):.4f}, "
        f"violations={result.violation_type or 'none'}"
    )[:1024]
    image.PatientName = ""
    image.PatientID = ""
    image.PatientBirthDate = ""
    image.PatientSex = ""
    image.StudyInstanceUID = safe_uid(result.study_uid)
    image.SeriesInstanceUID = series_uid or generate_uid()
    image.StudyID = ""
    image.SeriesNumber = VISUALIZATION_SERIES_NUMBER
    image.InstanceNumber = instance_number
    image.PatientOrientation = ""
    image.BurnedInAnnotation = "YES"
    image.SamplesPerPixel = 3
    image.PhotometricInterpretation = "RGB"
    image.PlanarConfiguration = 0
    image.Rows = rows
    image.Columns = columns
    image.BitsAllocated = 8
    image.BitsStored = 8
    image.HighBit = 7
    image.PixelRepresentation = 0
    image.PixelData = pixels.tobytes()
    # The explained source image (General Image module, Source Image Sequence).
    source_image = Dataset()
    source_image.ReferencedSOPClassUID = safe_uid(getattr(source, "SOPClassUID", ""))
    source_image.ReferencedSOPInstanceUID = safe_uid(
        getattr(source, "SOPInstanceUID", "") or result.image_uid
    )
    image.SourceImageSequence = [source_image]
    image.save_as(output, enforce_file_format=True)
    return output
