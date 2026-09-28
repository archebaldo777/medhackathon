"""Robustness of reading and batch processing on DICOM variants seen in the wild."""

from __future__ import annotations

import io

import numpy as np
import pydicom
from pydicom.uid import RLELossless

from src.analyzer import Analyzer
from src.batch import discover_dicom, process_directory
from src.dicom_io import read_dicom

from conftest import make_dicom


def spine_pixels() -> np.ndarray:
    pixels = np.zeros((317, 300), dtype=np.uint16)
    pixels[:, 130:170] = 2000
    pixels[40:60, 20:40] = 4000
    return pixels


def _dataset(tmp_path, name="base.dcm", **tags):
    path = make_dicom(tmp_path / name, spine_pixels(), **tags)
    return path, pydicom.dcmread(path)


def test_rgb_image_is_converted_to_grayscale(tmp_path):
    path, dataset = _dataset(tmp_path)
    gray = dataset.pixel_array.astype(np.float32)
    rgb = np.stack([dataset.pixel_array.astype(np.uint8)] * 3, axis=-1)
    dataset.BitsAllocated, dataset.BitsStored, dataset.HighBit = 8, 8, 7
    dataset.SamplesPerPixel = 3
    dataset.PhotometricInterpretation = "RGB"
    dataset.PlanarConfiguration = 0
    dataset.PixelData = rgb.tobytes()
    dataset.save_as(tmp_path / "rgb.dcm", enforce_file_format=True)

    image = read_dicom(tmp_path / "rgb.dcm")
    assert image.pixels.shape == gray.shape
    assert image.photometric == "MONOCHROME2"


def test_multiframe_uses_first_frame(tmp_path):
    path, dataset = _dataset(tmp_path)
    frame = dataset.pixel_array
    dataset.NumberOfFrames = 2
    dataset.PixelData = np.stack([frame, np.zeros_like(frame)]).tobytes()
    dataset.save_as(tmp_path / "multi.dcm", enforce_file_format=True)
    image = read_dicom(tmp_path / "multi.dcm")
    assert image.pixels.shape == frame.shape
    assert np.array_equal(image.pixels, frame.astype(np.float32))


def test_compressed_transfer_syntax_is_decoded(tmp_path):
    path, dataset = _dataset(tmp_path)
    dataset.compress(RLELossless)
    dataset.save_as(tmp_path / "rle.dcm", enforce_file_format=True)
    assert np.array_equal(read_dicom(tmp_path / "rle.dcm").pixels, read_dicom(path).pixels)


def test_bad_image_in_a_study_does_not_change_the_others(tmp_path):
    """A colour copy inside the same study used to poison the pooled features."""
    study = pydicom.uid.generate_uid()
    source = tmp_path / "in"
    source.mkdir()
    make_dicom(source / "a.dcm", spine_pixels(), StudyInstanceUID=study)
    alone = Analyzer().analyze(source / "a.dcm")[0].quality_prob

    dataset = pydicom.dcmread(source / "a.dcm")
    dataset.SOPInstanceUID = pydicom.uid.generate_uid()
    rgb = np.stack([np.clip(dataset.pixel_array // 16, 0, 255).astype(np.uint8)] * 3, axis=-1)
    dataset.BitsAllocated, dataset.BitsStored, dataset.HighBit = 8, 8, 7
    dataset.SamplesPerPixel, dataset.PhotometricInterpretation, dataset.PlanarConfiguration = 3, "RGB", 0
    dataset.PixelData = rgb.tobytes()
    dataset.save_as(source / "b.dcm", enforce_file_format=True)

    results = {r.path_to_study: r for r in process_directory(source, tmp_path / "out", visualizations=False)}
    assert abs(results["a.dcm"].quality_prob - alone) < 0.05


def test_missing_uids_are_reported_empty(tmp_path):
    path, dataset = _dataset(tmp_path)
    del dataset.StudyInstanceUID
    del dataset.SOPInstanceUID
    dataset.save_as(tmp_path / "no_uid.dcm", enforce_file_format=False)
    results = process_directory(tmp_path / "no_uid.dcm", tmp_path / "out", visualizations=False)
    assert results[0].processing_status == "Success"
    assert results[0].study_uid == "" and results[0].image_uid == ""


def test_legacy_file_without_preamble_or_extension_is_found(tmp_path):
    path, dataset = _dataset(tmp_path)
    buffer = io.BytesIO()
    pydicom.dcmwrite(buffer, dataset, enforce_file_format=False)
    (tmp_path / "IMG0001").write_bytes(buffer.getvalue()[132:])  # no preamble, no DICM
    (tmp_path / "README").write_text("not an image")
    path.unlink()
    assert [p.name for p in discover_dicom(tmp_path)] == ["IMG0001"]


def test_time_of_processing_is_per_image_not_cumulative(tmp_path):
    source = tmp_path / "in"
    source.mkdir()
    for index in range(6):
        make_dicom(source / f"{index}.dcm", spine_pixels())
    results = process_directory(source, tmp_path / "out", visualizations=True)
    times = [r.time_of_processing for r in results]
    # Cumulative timing made the first file carry the reading time of the whole batch.
    assert max(times) < 3 * min(times) + 0.5


def test_derived_dicom_objects_reference_the_source_image(tmp_path):
    source = tmp_path / "in"
    source.mkdir()
    path = make_dicom(source / "a.dcm", spine_pixels())
    original = pydicom.dcmread(path)
    results = process_directory(source, tmp_path / "out", visualizations=True)
    artifacts = results[0].artifacts

    report = pydicom.dcmread(tmp_path / "out" / artifacts["dicom_sr"])
    evidence = report.CurrentRequestedProcedureEvidenceSequence[0]
    assert evidence.StudyInstanceUID == original.StudyInstanceUID
    series = evidence.ReferencedSeriesSequence[0]
    assert series.SeriesInstanceUID == original.SeriesInstanceUID
    assert series.ReferencedSOPSequence[0].ReferencedSOPInstanceUID == original.SOPInstanceUID

    visual = pydicom.dcmread(tmp_path / "out" / artifacts["dicom_visualization"])
    assert visual.SourceImageSequence[0].ReferencedSOPInstanceUID == original.SOPInstanceUID
    assert visual.StudyInstanceUID == original.StudyInstanceUID
