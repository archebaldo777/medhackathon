from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pydicom

from src.batch import process_directory
from src.model import DECISION_POINT, align_to_threshold
from src.review import ReviewError, apply_review

from conftest import make_dicom

TEST_SPINE = Path(__file__).resolve().parents[1] / "dataset" / "for tests" / "CR000000_ПОП.dcm"


def _spine_study(tmp_path):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    pixels = np.zeros((317, 300), dtype=np.uint16)
    pixels[:, 130:170] = 2000
    make_dicom(input_dir / "spine.dcm", pixels)
    return input_dir


def test_alignment_sends_every_threshold_to_the_decision_point():
    for threshold in (0.08, 0.3, 0.5, 0.85):
        assert align_to_threshold(threshold, threshold) == DECISION_POINT
        assert align_to_threshold(threshold * 0.99, threshold) < DECISION_POINT
        assert align_to_threshold(0.0, threshold) == 0.0
        assert align_to_threshold(1.0, threshold) == 1.0
    # Monotone, so ranking within a head is preserved.
    values = [align_to_threshold(p, 0.2) for p in np.linspace(0, 1, 51)]
    assert values == sorted(values)


def test_exported_class_agrees_with_exported_probability(tmp_path):
    output = tmp_path / "output"
    results = process_directory(_spine_study(tmp_path), output, visualizations=True)
    for result in results:
        if result.processing_status != "Success":
            continue
        assert result.quality_class == int(result.quality_prob >= DECISION_POINT)
        assert result.decision_threshold == DECISION_POINT
        assert set(result.head_scores) >= {"spine_quality", "spine_axis_deviation"}
        assert "field_height_mm" in result.measurements
        assert "axis_angle_deg" in result.measurements
        for item in result.recommendations:
            assert {"criterion", "measured", "action"} <= set(item)


def test_visualization_is_saved_as_dicom_in_the_source_study(tmp_path):
    input_dir = _spine_study(tmp_path)
    source_uid = pydicom.dcmread(input_dir / "spine.dcm").StudyInstanceUID
    output = tmp_path / "output"
    [result] = process_directory(input_dir, output, visualizations=True)
    relative = result.artifacts["dicom_visualization"]
    image = pydicom.dcmread(output / relative)
    assert image.StudyInstanceUID == source_uid
    assert image.SOPClassUID == "1.2.840.10008.5.1.4.1.1.7"
    assert image.SamplesPerPixel == 3
    assert image.PhotometricInterpretation == "RGB"
    assert str(image.PatientName) == ""
    assert image.pixel_array.shape[2] == 3


def _official_spine(tmp_path):
    input_dir = tmp_path / "official"
    input_dir.mkdir()
    shutil.copy(TEST_SPINE, input_dir / "spine.dcm")
    return input_dir


def test_review_marks_sr_verified_and_keeps_model_output(tmp_path):
    input_dir = _official_spine(tmp_path)
    output = tmp_path / "output"
    results = process_directory(input_dir, output, visualizations=False)
    sr_path = output / results[0].artifacts["dicom_sr"]
    assert pydicom.dcmread(sr_path).VerificationFlag == "UNVERIFIED"
    model_class = results[0].quality_class
    csv_before = (output / "results.csv").read_bytes()

    try:
        apply_review(results, 0, output, input_dir / "spine.dcm", "confirmed", "  ")
    except ReviewError:
        pass
    else:
        raise AssertionError("review without a reviewer was accepted")

    apply_review(results, 0, output, input_dir / "spine.dcm", "rejected", "Петров", "артефакт")
    sr = pydicom.dcmread(sr_path)
    assert sr.VerificationFlag == "VERIFIED"
    assert str(sr.VerifyingObserverSequence[0].VerifyingObserverName) == "Петров"
    texts = [item.TextValue for item in sr.ContentSequence if item.ValueType == "TEXT"]
    assert "Specialist rejected the AI conclusion" in texts
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["results"][0]["review"]["decision"] == "rejected"
    assert manifest["results"][0]["review"]["model_quality_class"] == model_class
    assert manifest["results"][0]["markup"]["status"] == "accepted"
    # The ТЗ report keeps what the model said.
    assert (output / "results.csv").read_bytes() == csv_before


def test_specialist_edit_moves_markup_and_rewrites_presentation_state(tmp_path):
    input_dir = _official_spine(tmp_path)
    output = tmp_path / "output"
    results = process_directory(input_dir, output, visualizations=False)
    markup = results[0].markup
    gsps_path = output / results[0].artifacts["dicom_markup"]
    before = pydicom.dcmread(gsps_path)
    assert before.Modality == "PR"
    assert before.ContentDescription.startswith("Автоматическая")

    shapes = [{"label": s["label"], "points": [list(p) for p in s["points"]]} for s in markup["shapes"]]
    top = next(s for s in shapes if s["label"] == "T12/L1")
    for point in top["points"]:
        point[1] -= 5  # the specialist lifts the upper L1 border by 5 pixels
    apply_review(
        results, 0, output, input_dir / "spine.dcm", "confirmed", "Иванова",
        markup_decision="edited", markup_shapes=shapes,
    )
    edited = results[0].markup
    assert edited["status"] == "edited" and edited["source"] == "specialist"
    moved = next(s for s in edited["shapes"] if s["label"] == "T12/L1")
    original = next(s for s in markup["shapes"] if s["label"] == "T12/L1")
    assert moved["points"][0][1] == round(original["points"][0][1] - 5, 2)
    assert "automatic_shapes" in edited
    state = pydicom.dcmread(gsps_path)
    assert state.ContentDescription == "Разметка исправлена специалистом"
    assert str(state.ContentCreatorName) == "Иванова"

    apply_review(
        results, 0, output, input_dir / "spine.dcm", "confirmed", "Иванова",
        markup_decision="rejected",
    )
    assert not gsps_path.exists()
    assert "dicom_markup" not in results[0].artifacts


def test_markup_edit_cannot_invent_or_reshape_elements(tmp_path):
    input_dir = _official_spine(tmp_path)
    results = process_directory(input_dir, tmp_path / "output", visualizations=False)
    for bad in (
        [{"label": "L5/S1", "points": [[1, 1], [2, 2]]}],
        [{"label": "T12/L1", "points": [[1, 1]]}],
    ):
        try:
            apply_review(
                results, 0, tmp_path / "output", input_dir / "spine.dcm", "confirmed", "X",
                markup_decision="edited", markup_shapes=bad,
            )
        except ReviewError:
            continue
        raise AssertionError(f"invalid markup edit accepted: {bad}")
