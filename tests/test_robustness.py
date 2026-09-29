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


def test_failure_row_keeps_dicom_identifiers(tmp_path):
    """A file with a readable header but broken pixels is still matched by its UIDs."""
    source = tmp_path / "in"
    source.mkdir()
    path = make_dicom(source / "broken.dcm", spine_pixels())
    dataset = pydicom.dcmread(path)
    dataset.PixelData = dataset.PixelData[:100]
    dataset.save_as(path, enforce_file_format=True)
    (source / "garbage.dcm").write_bytes(b"not a dicom")

    results = {r.path_to_study: r for r in process_directory(source, tmp_path / "out", visualizations=False)}
    broken = results["broken.dcm"]
    assert broken.processing_status == "Failure"
    assert broken.study_uid == dataset.StudyInstanceUID
    assert broken.image_uid == dataset.SOPInstanceUID
    garbage = results["garbage.dcm"]
    assert garbage.processing_status == "Failure"
    assert garbage.study_uid == "" and garbage.image_uid == ""


def test_error_column_carries_failure_reason(tmp_path):
    import csv

    source = tmp_path / "in"
    source.mkdir()
    make_dicom(source / "ok.dcm", spine_pixels())
    (source / "garbage.dcm").write_bytes(b"not a dicom")
    process_directory(source, tmp_path / "out", visualizations=False)
    with (tmp_path / "out" / "results.csv").open(encoding="utf-8-sig") as handle:
        rows = {row["path_to_study"]: row for row in csv.DictReader(handle)}
    assert rows["garbage.dcm"]["processing_status"] == "Failure"
    assert rows["garbage.dcm"]["error"]
    assert rows["ok.dcm"]["error"] == ""


def test_quality_class_without_type_is_explained():
    from src.analyzer import positioning_recommendations
    from src.model import ModelResult

    predicted = ModelResult(
        region="spine",
        quality_class=1,
        quality_probability=0.7,
        violation_keys=(),
        probabilities={
            "spine_quality": 0.9,
            "spine_positioning": 0.1,
            "spine_axis_deviation": 0.19,
            "foreign_object_or_artifact": 0.2,
        },
        model_version="test",
        decision_threshold=0.5,
        head_thresholds={
            "spine_quality": 0.3,
            "spine_positioning": 0.2,
            "spine_axis_deviation": 0.2,
            "foreign_object_or_artifact": 0.7,
        },
    )
    items = positioning_recommendations(predicted, {}, {})
    assert items[-1]["criterion"] == "Тип нарушения не уточнён"
    assert "Не выравнена ось" in items[-1]["measured"] or "ось" in items[-1]["measured"]
