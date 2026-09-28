from __future__ import annotations

import io
import zipfile

import numpy as np
from fastapi.testclient import TestClient

from src.api import app

from conftest import make_dicom


def test_health_and_analyze(tmp_path):
    source = make_dicom(
        tmp_path / "sample.dcm",
        np.pad(np.full((250, 40), 1500, dtype=np.uint16), ((35, 32), (130, 130))),
    )
    client = TestClient(app)
    health = client.get("/health")
    assert health.status_code == 200
    assert health.json()["status"] == "ok"

    with source.open("rb") as handle:
        response = client.post(
            "/v1/analyze",
            files=[("files", ("sample.dcm", handle, "application/dicom"))],
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/zip"
    with zipfile.ZipFile(io.BytesIO(response.content)) as result:
        assert {"results.csv", "results.xlsx", "manifest.json"}.issubset(result.namelist())


def test_model_info_exposes_submission_vocabulary_and_calibration():
    response = TestClient(app).get("/v1/model")
    assert response.status_code == 200
    payload = response.json()
    assert payload["supported_regions"] == [
        "Поясничный отдел позвоночника",
        "Проксимальный отдел бедра",
    ]
    assert payload["scanner_calibration"]["pixel_spacing_y_mm"] == 1.05
    assert payload["scanner_calibration"]["pixel_spacing_x_mm"] == 0.6
    assert payload["violation_types"]["Проксимальный отдел бедра"] == [
        "Некорректная укладка",
        "Некорректная область интереса",
    ]


def test_web_job_returns_result_preview_and_download(tmp_path):
    source = make_dicom(
        tmp_path / "job.dcm",
        np.pad(np.full((250, 40), 1500, dtype=np.uint16), ((35, 32), (130, 130))),
    )
    client = TestClient(app)
    with source.open("rb") as handle:
        response = client.post(
            "/v1/jobs",
            files=[("files", (source.name, handle, "application/dicom"))],
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["success"] == 1
    assert payload["results"][0]["anatomical_region"] == "Поясничный отдел позвоночника"
    preview = client.get(payload["results"][0]["preview_url"])
    assert preview.status_code == 200
    assert preview.headers["content-type"] == "image/png"
    # The overlay and the plain image share one frame, so the viewer never jumps.
    from PIL import Image

    overlay = client.get(payload["results"][0]["overlay_url"])
    plain = client.get(f"/v1/jobs/{payload['job_id']}/image/0")
    assert overlay.status_code == 200 and plain.status_code == 200
    assert Image.open(io.BytesIO(overlay.content)).size == Image.open(io.BytesIO(plain.content)).size
    download = client.get(payload["download_url"])
    assert download.status_code == 200
    assert download.headers["content-type"] == "application/zip"


def test_brand_assets_are_served_and_nothing_else_is():
    client = TestClient(app)
    favicon = client.get("/favicon.ico")
    assert favicon.status_code == 200
    assert favicon.headers["content-type"] == "image/x-icon"
    for name in ("logo-mark.png", "favicon-32.png", "apple-touch-icon.png", "demo-spine.png", "demo-heat.png"):
        response = client.get(f"/static/{name}")
        assert response.status_code == 200, name
        assert response.headers["content-type"] == "image/png"
    model = client.get("/static/spine-3d.bin")
    assert model.status_code == 200
    assert model.content[:4] == b"SPN1"
    assert client.get("/static/index.html").status_code == 404
    assert client.get("/static/..%2Fapi.py").status_code == 404
    assert "Эвектио" in client.get("/").text


def test_non_dicom_upload_is_rejected(tmp_path):
    picture = tmp_path / "photo.jpg"
    picture.write_bytes(b"not a dicom")
    with picture.open("rb") as handle:
        response = TestClient(app).post(
            "/v1/jobs", files=[("files", ("photo.jpg", handle, "image/jpeg"))]
        )
    assert response.status_code == 415


def test_specialist_review_verifies_the_sr_and_updates_the_archive(tmp_path):
    """ТЗ 2.6: the specialist confirms or rejects the result through the API."""
    import json

    import pydicom

    source = make_dicom(
        tmp_path / "review.dcm",
        np.pad(np.full((250, 40), 1500, dtype=np.uint16), ((35, 32), (130, 130))),
    )
    client = TestClient(app)
    with source.open("rb") as handle:
        job = client.post(
            "/v1/jobs", files=[("files", ("review.dcm", handle, "application/dicom"))]
        ).json()
    result = job["results"][0]
    assert result["review"] == {}

    rejected = client.post(
        f"/v1/jobs/{job['job_id']}/review/0",
        json={"decision": "maybe", "reviewer": "Иванова"},
    )
    assert rejected.status_code == 422

    response = client.post(
        f"/v1/jobs/{job['job_id']}/review/0",
        json={
            "decision": "confirmed",
            "reviewer": "Иванова А.А.",
            "comment": "Согласна",
            "markup_decision": "accepted",
        },
    )
    assert response.status_code == 200
    review = response.json()["result"]["review"]
    assert review["decision"] == "confirmed"
    assert review["reviewer"] == "Иванова А.А."

    archive = client.get(job["download_url"])
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        manifest = json.loads(bundle.read("manifest.json"))
        assert manifest["results"][0]["review"]["decision"] == "confirmed"
        sr_name = next(name for name in bundle.namelist() if name.startswith("dicom_sr/"))
        sr = pydicom.dcmread(io.BytesIO(bundle.read(sr_name)))
        assert sr.VerificationFlag == "VERIFIED"
        assert str(sr.VerifyingObserverSequence[0].VerifyingObserverName) == "Иванова А.А."
        assert any(name.startswith("dicom_viz/") for name in bundle.namelist())
