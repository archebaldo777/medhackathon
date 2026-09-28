from __future__ import annotations

from pathlib import Path

import pydicom

from src.dicom_gsps import GRAYSCALE_SOFTCOPY_PRESENTATION_STATE, save_markup_gsps
from src.dicom_io import anatomical_region, normalize_pixels, read_dicom
from src.markup import SPINE_DISC_LABELS, build_markup, edited_markup

TESTS = Path(__file__).resolve().parents[1] / "dataset" / "for tests"


def _markup(name: str):
    dicom = read_dicom(TESTS / name)
    normalized = normalize_pixels(dicom.pixels)
    region = anatomical_region(normalized)
    return build_markup(normalized, region), normalized.shape


def test_spine_markup_orders_l1_to_l4_from_the_top():
    markup, _ = _markup("CR000000_ПОП.dcm")
    discs = [s for s in markup.shapes if s["label"] in SPINE_DISC_LABELS]
    assert [s["label"] for s in discs] == list(SPINE_DISC_LABELS)
    heights = [s["points"][0][1] for s in discs]
    assert heights == sorted(heights), "T12/L1 must be above L4/L5"
    vertebrae = [s for s in markup.derived if s["label"] in {"L1", "L2", "L3", "L4"}]
    assert [s["label"] for s in vertebrae] == ["L1", "L2", "L3", "L4"]
    # Like the densitometer, the L1–L4 boxes share their left and right borders.
    lefts = {round(min(p[0] for p in s["points"]), 1) for s in vertebrae}
    assert len(lefts) == 1
    edges = [s for s in markup.derived if s.get("style") == "edge"]
    assert len(edges) == 2 and all(len(s["points"]) > 20 for s in edges)
    assert "curve_deviation_mm" in markup.as_dict()["measurements"]
    criteria = {c["criterion"] for c in markup.checks}
    assert "В поле видны верхние края подвздошных костей" in criteria


def test_hip_markup_checks_field_margins_in_millimetres():
    for name in ("CR000000_ППОБ.dcm", "CR000001_ЛПОБ.dcm"):
        markup, shape = _markup(name)
        labels = {s["label"] for s in markup.shapes}
        assert "Шейка бедра" in labels
        derived = {s["label"] for s in markup.derived}
        assert {"Контур кости", "Линия вертела", "Головка бедра"} <= derived
        margins = {c["criterion"]: c for c in markup.checks}
        assert margins["Поле выше области интереса"]["required"] == 30.0
        assert margins["Поле ниже области интереса"]["required"] == 30.0
        assert margins["Поле от бокового края"]["required"] == 20.0
        for item in markup.shapes:
            for x, y in item["points"]:
                assert 0 <= x < shape[1] and 0 <= y < shape[0]


def test_moving_the_roi_recomputes_the_margin_check():
    markup, shape = _markup("CR000000_ППОБ.dcm")
    roi = next(s for s in markup.shapes if s["label"].startswith("Область интереса"))
    moved = [{"label": roi["label"], "points": [[x, 2.0 if i < 2 else y] for i, (x, y) in enumerate(roi["points"])]}]
    edited = edited_markup(markup.as_dict(), moved, shape)
    above = next(c for c in edited.checks if c["criterion"] == "Поле выше области интереса")
    assert above["ok"] is False
    assert edited.source == "specialist"
    # The trochanteric line is rebuilt from the moved box; the bone edge stays.
    line = next(s for s in edited.derived if s["label"] == "Линия вертела")
    assert line["points"][0][1] == 2.0
    assert any(s["label"] == "Контур кости" for s in edited.derived)


def test_markup_is_written_as_presentation_state_over_the_source(tmp_path):
    markup, shape = _markup("CR000000_ПОП.dcm")
    payload = markup.as_dict() | {"image_shape": list(shape), "status": "proposed"}
    path = save_markup_gsps(TESTS / "CR000000_ПОП.dcm", payload, tmp_path / "markup.dcm")
    state = pydicom.dcmread(path)
    source = pydicom.dcmread(TESTS / "CR000000_ПОП.dcm")
    assert state.SOPClassUID == GRAYSCALE_SOFTCOPY_PRESENTATION_STATE
    assert state.StudyInstanceUID == source.StudyInstanceUID
    image = state.ReferencedSeriesSequence[0].ReferencedImageSequence[0]
    assert image.ReferencedSOPInstanceUID == source.SOPInstanceUID
    graphics = state.GraphicAnnotationSequence[0].GraphicObjectSequence
    assert len(graphics) == len(payload["shapes"]) + len(payload["derived"])
    assert str(state.PatientName) == ""
