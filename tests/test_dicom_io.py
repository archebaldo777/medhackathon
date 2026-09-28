from __future__ import annotations

import numpy as np
import pytest

from src.constants import LEFT_HIP, RIGHT_HIP, SPINE
from src.dicom_io import (
    anatomical_region,
    distance_cm_to_pixels,
    normalize_pixels,
    read_dicom,
)

from conftest import make_dicom


def test_read_and_normalize_dicom(tmp_path):
    pixels = np.arange(300 * 300, dtype=np.uint16).reshape(300, 300)
    source = make_dicom(tmp_path / "image.dcm", pixels)
    dicom = read_dicom(source)
    normalized = normalize_pixels(dicom.pixels)
    assert dicom.rows == 300
    assert dicom.columns == 300
    assert dicom.pixel_spacing_y_mm == 1.05
    assert dicom.pixel_spacing_x_mm == 0.6
    assert dicom.pixel_spacing_source == "scanner_default"
    assert normalized.min() == 0
    assert normalized.max() == 1
    assert anatomical_region(normalized) == SPINE


def test_hip_laterality_from_lower_femoral_shaft():
    right = np.zeros((290, 280), dtype=np.float32)
    right[120:, 55:100] = 1
    left = np.fliplr(right)
    assert anatomical_region(right) == RIGHT_HIP
    assert anatomical_region(left) == LEFT_HIP


def test_dicom_spacing_has_priority_over_scanner_default(tmp_path):
    source = make_dicom(tmp_path / "calibrated.dcm", np.zeros((20, 20), dtype=np.uint16))
    import pydicom

    dataset = pydicom.dcmread(source)
    dataset.PixelSpacing = [0.8, 0.7]
    dataset.save_as(source, enforce_file_format=True)

    dicom = read_dicom(source)
    assert dicom.pixel_spacing_y_mm == 0.8
    assert dicom.pixel_spacing_x_mm == 0.7
    assert dicom.pixel_spacing_source == "PixelSpacing"


def test_physical_distances_use_axis_specific_spacing():
    assert distance_cm_to_pixels(3.0, 1.05) == pytest.approx(28.5714286)
    assert distance_cm_to_pixels(2.0, 0.6) == pytest.approx(33.3333333)
