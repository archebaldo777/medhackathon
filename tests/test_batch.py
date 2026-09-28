from __future__ import annotations

import csv
import zipfile

import numpy as np
import pydicom

from src.batch import UnsafeArchiveError, process_directory, safe_extract_zip
from src.constants import REPORT_COLUMNS

from conftest import make_dicom


def test_batch_writes_required_reports(tmp_path):
    assert REPORT_COLUMNS == (
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
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pixels = np.zeros((317, 300), dtype=np.uint16)
    pixels[:, 130:170] = 2000
    make_dicom(input_dir / "valid.dcm", pixels)
    (input_dir / "broken.dcm").write_bytes(b"not a dicom")

    output = tmp_path / "output"
    results = process_directory(input_dir, output, visualizations=False)
    assert len(results) == 2
    assert {item.processing_status for item in results} == {"Success", "Failure"}
    with (output / "results.csv").open(encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        assert tuple(reader.fieldnames or ()) == REPORT_COLUMNS
        rows = list(reader)
        assert len(rows) == 2
        successful = next(row for row in rows if row["processing_status"] == "Success")
        assert successful["anatomical_region"] == "Поясничный отдел позвоночника"
        assert 0.0 <= float(successful["quality_prob"]) <= 1.0
        allowed = {
            "Некорректная укладка",
            "Не выравнена ось позвоночника",
            "Присутствуют посторонние предметы",
        }
        assert not successful["violation_type"] or set(
            successful["violation_type"].split(";")
        ) <= allowed
    assert (output / "results.xlsx").exists()
    assert (output / "manifest.json").exists()
    sr_files = list((output / "dicom_sr").glob("*.dcm"))
    assert len(sr_files) == 1
    sr = pydicom.dcmread(sr_files[0])
    assert sr.Modality == "SR"
    assert sr.CompletionFlag == "COMPLETE"
    assert sr.ContentSequence[-1].ValueType == "IMAGE"


def test_zip_slip_is_rejected(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("../escape.dcm", b"bad")
    try:
        safe_extract_zip(archive, tmp_path / "extract")
    except UnsafeArchiveError:
        pass
    else:
        raise AssertionError("unsafe archive was accepted")


def test_batch_pools_same_study_and_region(tmp_path):
    input_dir = tmp_path / "series"
    input_dir.mkdir()
    first = make_dicom(
        input_dir / "frame-a.dcm",
        np.pad(np.full((220, 30), 1800, dtype=np.uint16), ((40, 40), (135, 135))),
    )
    second = make_dicom(
        input_dir / "frame-b.dcm",
        np.pad(np.full((180, 55), 1300, dtype=np.uint16), ((60, 60), (122, 123))),
    )
    shared_uid = pydicom.dcmread(first).StudyInstanceUID
    second_dataset = pydicom.dcmread(second)
    second_dataset.StudyInstanceUID = shared_uid
    second_dataset.save_as(second, enforce_file_format=True)

    results = process_directory(input_dir, tmp_path / "pooled", visualizations=False)
    assert len(results) == 2
    assert results[0].model_version == "dxa-quality-criteria-4.0"
    assert results[0].quality_prob == results[1].quality_prob
