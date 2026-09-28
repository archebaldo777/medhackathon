from __future__ import annotations

import asyncio
import csv
import io
import zipfile

import numpy as np
from fastapi.testclient import TestClient

from src import api
from src.batch import discover_dicom, process_directory, safe_extract_zip
from src.constants import LEFT_HIP, RIGHT_HIP
from src.projection import (
    FRONTAL,
    LATERAL,
    detect_projection,
    laterality_from_tags,
    projection_from_tags,
)

from conftest import make_dicom


def spine_pixels() -> np.ndarray:
    pixels = np.zeros((317, 300), dtype=np.uint16)
    pixels[:, 130:170] = 2000
    return pixels


def hip_pixels(shaft_on_left: bool) -> np.ndarray:
    pixels = np.zeros((291, 280), dtype=np.uint16)
    columns = slice(40, 100) if shaft_on_left else slice(180, 240)
    pixels[160:, columns] = 2500
    return pixels


# --- projection (ТЗ 2.2) ---------------------------------------------------------


def test_projection_from_view_position():
    assert projection_from_tags({"ViewPosition": "PA"}).code == FRONTAL
    assert projection_from_tags({"ViewPosition": "LL"}).code == LATERAL
    assert projection_from_tags({"ViewPosition": "AP"}).source == "dicom:ViewPosition"


def test_projection_from_patient_orientation():
    # Organiser files: rows towards the patient's left, columns towards the feet.
    frontal = projection_from_tags({"ViewPosition": "", "PatientOrientation": ["L", "F"]})
    assert frontal.code == FRONTAL and frontal.source == "dicom:PatientOrientation"
    assert projection_from_tags({"PatientOrientation": "A\\F"}).code == LATERAL
    assert projection_from_tags({"PatientOrientation": ["P", "H"]}).code == LATERAL
    # An axial plane is not a DXA projection and is not guessed.
    assert projection_from_tags({"PatientOrientation": ["L", "P"]}) is None


def test_projection_from_descriptions_and_protocol_fallback():
    assert projection_from_tags({"SeriesDescription": "Lateral Spine"}).code == LATERAL
    assert projection_from_tags({"ProtocolName": "AP Spine L1-L4"}).code == FRONTAL
    assert projection_from_tags({"StudyDescription": "DXA Обследование"}) is None
    fallback = detect_projection({})
    assert fallback.code == FRONTAL and fallback.source == "protocol"


def test_laterality_from_tags():
    assert laterality_from_tags({"Laterality": "R"}) == RIGHT_HIP
    assert laterality_from_tags({"ImageLaterality": "L"}) == LEFT_HIP
    assert laterality_from_tags({"Laterality": ""}) is None


def test_report_contains_projection_and_hip_side(tmp_path):
    source = tmp_path / "in"
    source.mkdir()
    make_dicom(source / "spine.dcm", spine_pixels(), PatientOrientation=["L", "F"])
    make_dicom(source / "right.dcm", hip_pixels(shaft_on_left=True), PatientOrientation=["L", "F"])
    make_dicom(source / "left.dcm", hip_pixels(shaft_on_left=False), PatientOrientation=["L", "F"])
    make_dicom(source / "lateral.dcm", spine_pixels(), PatientOrientation=["A", "F"])
    results = {r.path_to_study: r for r in process_directory(source, tmp_path / "out", visualizations=False)}

    assert results["spine.dcm"].projection == "Прямая (AP)"
    assert results["spine.dcm"].projection_source == "dicom:PatientOrientation"
    assert results["right.dcm"].anatomical_region == "Проксимальный отдел бедра (правый)"
    assert results["left.dcm"].anatomical_region == "Проксимальный отдел бедра (левый)"
    lateral = results["lateral.dcm"]
    assert lateral.projection_code == LATERAL
    assert lateral.review_priority == "high"
    assert "Боковая проекция" in lateral.recommendation

    with (tmp_path / "out" / "results.csv").open(encoding="utf-8-sig") as handle:
        rows = {row["path_to_study"]: row for row in csv.DictReader(handle)}
    assert rows["lateral.dcm"]["projection"] == "Боковая (LAT)"


# --- input discovery and API robustness (ТЗ 2.7) -------------------------------------


def test_dicom_without_extension_is_found(tmp_path):
    make_dicom(tmp_path / "IM000001", spine_pixels())
    (tmp_path / "notes.txt").write_text("not an image")
    (tmp_path / "DICOMDIR").write_bytes(b"\0" * 128 + b"DICM")
    (tmp_path / "__MACOSX").mkdir()
    (tmp_path / "__MACOSX" / "._IM000001.dcm").write_bytes(b"resource fork")
    assert [path.name for path in discover_dicom(tmp_path)] == ["IM000001"]
    results = process_directory(tmp_path, tmp_path / "out", visualizations=False)
    assert [r.processing_status for r in results] == ["Success"]


def test_api_accepts_dicom_without_extension(tmp_path):
    source = make_dicom(tmp_path / "IM000001", spine_pixels())
    response = TestClient(api.app).post(
        "/v1/jobs", files=[("files", ("IM000001", source.read_bytes(), "application/octet-stream"))]
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["path_to_study"] == "IM000001"


def test_single_zip_keeps_paths_from_the_archive(tmp_path):
    source = make_dicom(tmp_path / "a.dcm", spine_pixels())
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.write(source, "study-1/series-2/a.dcm")
    response = TestClient(api.app).post(
        "/v1/jobs", files=[("files", ("batch.zip", buffer.getvalue(), "application/zip"))]
    )
    assert response.status_code == 200
    assert response.json()["results"][0]["path_to_study"] == "study-1/series-2/a.dcm"


def test_broken_zip_is_a_client_error_and_leaves_no_files(tmp_path, monkeypatch):
    monkeypatch.setattr(api.tempfile, "tempdir", str(tmp_path))
    response = TestClient(api.app).post(
        "/v1/analyze", files=[("files", ("broken.zip", b"garbage", "application/zip"))]
    )
    assert response.status_code == 400
    assert "ZIP" in response.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_zip_without_dicom_is_a_client_error(tmp_path):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("readme.txt", "hello")
    response = TestClient(api.app).post(
        "/v1/analyze", files=[("files", ("empty.zip", buffer.getvalue(), "application/zip"))]
    )
    assert response.status_code == 400
    assert "/tmp" not in response.json()["detail"]


def test_analysis_runs_off_the_event_loop(tmp_path, monkeypatch):
    """CPU-bound analysis must not block /health and the UI while a batch runs."""
    seen = {}
    real = api.process_directory

    def spy(*args, **kwargs):
        try:
            asyncio.get_running_loop()
            seen["on_event_loop"] = True
        except RuntimeError:
            seen["on_event_loop"] = False
        return real(*args, **kwargs)

    monkeypatch.setattr(api, "process_directory", spy)
    source = make_dicom(tmp_path / "a.dcm", spine_pixels())
    response = TestClient(api.app).post(
        "/v1/analyze", files=[("files", ("a.dcm", source.read_bytes(), "application/dicom"))]
    )
    assert response.status_code == 200
    assert seen == {"on_event_loop": False}


def _legacy_zip(path, name_bytes: bytes, payload: bytes) -> None:
    """A ZIP whose member name is stored as raw bytes without the UTF-8 flag."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as bundle:
        bundle.writestr("PLACEHOLDER.dcm", payload)
    data = buffer.getvalue().replace(b"PLACEHOLDER.dcm", name_bytes.ljust(15, b"_")[:15])
    path.write_bytes(data)


def test_zip_with_cyrillic_names_without_utf8_flag(tmp_path):
    for encoding in ("utf-8", "cp866"):
        name = "ППОБ.dcm".encode(encoding)
        archive = tmp_path / f"{encoding}.zip"
        _legacy_zip(archive, name, b"x")
        extracted = safe_extract_zip(archive, tmp_path / encoding)
        assert extracted[0].name.startswith("ППОБ"), (encoding, extracted[0].name)
