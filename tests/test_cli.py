from __future__ import annotations

import json
import sys

import pytest

from src.cli import main


def test_cli_reports_missing_dicom_without_traceback(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        ["evectio", "analyze", str(tmp_path), "--output", str(tmp_path / "out")],
    )
    with pytest.raises(SystemExit) as stopped:
        main()
    assert stopped.value.code == 1
    payload = json.loads(capsys.readouterr().err)
    assert payload["status"] == "Failure"
    assert "DICOM-файлы не найдены" in payload["error"]
