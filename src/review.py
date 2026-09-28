"""Specialist confirmation of the conclusion and of the markup (ТЗ 2.6).

Two separate decisions are recorded, because the ТЗ asks for two things:

* the conclusion — does the specialist agree with the quality class and the
  violations the model reported (``confirmed`` / ``rejected``);
* the markup — the L1–L4 levels or the hip ROI and neck box that the service
  placed automatically, the same markup the densitometer draws and the
  specialist normally corrects (``accepted`` / ``edited`` / ``rejected``).

The decision is written back into every artefact that carries it:
``manifest.json``, the DICOM SR (VerificationFlag=VERIFIED with the verifying
observer) and the markup presentation state, which is rewritten with the edited
points or removed when the markup is rejected. ``results.csv``/``results.xlsx``
keep the model output in the format fixed by the ТЗ, so a specialist's decision
never silently replaces what the model said.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from .analyzer import AnalysisResult
from .dicom_gsps import save_markup_gsps
from .dicom_sr import save_dicom_sr
from .markup import edited_markup
from .reporting import write_manifest

DECISIONS = {"confirmed", "rejected"}
MARKUP_DECISIONS = {"accepted", "edited", "rejected"}


class ReviewError(ValueError):
    pass


def apply_review(
    results: list[AnalysisResult],
    index: int,
    output_dir: str | Path,
    source_path: str | Path,
    decision: str,
    reviewer: str,
    comment: str = "",
    markup_decision: str = "accepted",
    markup_shapes: list[dict[str, Any]] | None = None,
    organization: str = "",
) -> AnalysisResult:
    """Record a specialist decision for ``results[index]`` and refresh artefacts."""
    if not 0 <= index < len(results):
        raise ReviewError("Результат с таким номером не найден")
    result = results[index]
    if result.processing_status != "Success":
        raise ReviewError("Подтверждение доступно только для успешно обработанных исследований")
    if decision not in DECISIONS:
        raise ReviewError("decision должен быть confirmed или rejected")
    if markup_decision not in MARKUP_DECISIONS:
        raise ReviewError("markup_decision должен быть accepted, edited или rejected")
    reviewer = (reviewer or "").strip()
    if not reviewer:
        raise ReviewError("Укажите специалиста, подтверждающего результат")

    markup = dict(result.markup or {})
    if markup_decision == "edited":
        if not markup.get("shapes"):
            raise ReviewError("Автоматическая разметка отсутствует — исправлять нечего")
        try:
            updated = edited_markup(
                markup,
                markup_shapes or [],
                tuple(markup["image_shape"]),
                float(result.pixel_spacing_y_mm or 1.05),
                float(result.pixel_spacing_x_mm or 0.6),
            ).as_dict()
        except (KeyError, TypeError, ValueError) as exc:
            raise ReviewError(f"Разметка не принята: {exc}") from exc
        for key in ("image_shape", "pixel_spacing_y_mm", "pixel_spacing_x_mm"):
            updated[key] = markup.get(key)
        updated["automatic_shapes"] = markup.get("automatic_shapes", markup.get("shapes"))
        markup = updated
    if markup:
        markup["status"] = markup_decision
    result.markup = markup

    result.review = {
        "decision": decision,
        "reviewer": reviewer[:64],
        "organization": organization[:64],
        "comment": comment.strip()[:1000],
        "reviewed_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "model_quality_class": result.quality_class,
        "markup_decision": markup_decision,
    }

    output = Path(output_dir)
    gsps_relative = result.artifacts.get("dicom_markup")
    if gsps_relative:
        target = output / gsps_relative
        if markup_decision == "rejected":
            target.unlink(missing_ok=True)
            result.artifacts.pop("dicom_markup", None)
        else:
            save_markup_gsps(source_path, markup, target, creator=reviewer)
    sr_relative = result.artifacts.get("dicom_sr")
    if sr_relative:
        save_dicom_sr(source_path, result, output / sr_relative)
    write_manifest(results, output / "manifest.json")
    return result
