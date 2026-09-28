from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from pydicom import dcmread
from pydicom.dataset import Dataset, FileDataset, FileMetaDataset
from pydicom.uid import BasicTextSRStorage, ExplicitVRLittleEndian, generate_uid

from .analyzer import AnalysisResult


def _code(value: str, meaning: str, scheme: str = "99EVECTIO") -> Dataset:
    item = Dataset()
    item.CodeValue = value
    item.CodingSchemeDesignator = scheme
    item.CodeMeaning = meaning
    return item


def _text(name: str, meaning: str, value: str) -> Dataset:
    item = Dataset()
    item.RelationshipType = "CONTAINS"
    item.ValueType = "TEXT"
    item.ConceptNameCodeSequence = [_code(name, meaning)]
    item.TextValue = value
    return item


def safe_uid(value: object) -> str:
    """Keep a source UID whenever it can be written, so derived objects link to it.

    The supplied exports use UID components with leading zeros, which the
    standard forbids but PACS systems store as-is. Replacing such a UID would
    put the SR, the visual explanation and the markup into a different study
    than the image they describe, so it is kept verbatim; a new UID is generated
    only when the value is empty, too long or contains other characters.
    """
    candidate = str(value or "").strip().rstrip("\x00")
    valid = (
        bool(candidate)
        and len(candidate) <= 64
        and bool(re.fullmatch(r"[0-9]+(?:\.[0-9]+)*", candidate))
    )
    if valid:
        return candidate
    # Without a source UID there is nothing to link to: a fresh UID avoids merging
    # unrelated images into one study.
    return generate_uid(entropy_srcs=[candidate]) if candidate else generate_uid()


def safe_date(value: object) -> str:
    candidate = str(value or "")
    return candidate if re.fullmatch(r"\d{8}", candidate) else ""


def safe_time(value: object) -> str:
    candidate = str(value or "")
    return candidate if re.fullmatch(r"\d{2,6}(\.\d{1,6})?", candidate) else ""


def _dicom_datetime(value: str) -> str:
    try:
        moment = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        moment = datetime.now().astimezone()
    return moment.strftime("%Y%m%d%H%M%S")


def save_dicom_sr(
    source_path: str | Path,
    result: AnalysisResult,
    output_path: str | Path,
) -> Path:
    """Write a de-identified Basic Text SR linked to the source DXA image.

    Until a specialist has reviewed the result the report is UNVERIFIED. When
    ``result.review`` holds a decision the same report is written as VERIFIED
    with the verifying observer and the specialist's verdict on the conclusion
    and on the ROI markup (ТЗ 2.6).
    """
    review = result.review or {}
    source = dcmread(source_path, stop_before_pixels=True, force=True)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    sop_uid = generate_uid()
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = BasicTextSRStorage
    file_meta.MediaStorageSOPInstanceUID = sop_uid
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    file_meta.ImplementationClassUID = generate_uid()

    report = FileDataset(str(output), {}, file_meta=file_meta, preamble=b"\0" * 128)
    report.SpecificCharacterSet = "ISO_IR 192"
    report.SOPClassUID = BasicTextSRStorage
    report.SOPInstanceUID = sop_uid
    report.StudyInstanceUID = safe_uid(result.study_uid)
    report.SeriesInstanceUID = generate_uid()
    report.Modality = "SR"
    report.SeriesNumber = 900
    report.InstanceNumber = 1
    report.PatientName = ""
    report.PatientID = ""
    report.StudyDate = safe_date(getattr(source, "StudyDate", ""))
    report.StudyTime = safe_time(getattr(source, "StudyTime", ""))
    report.AccessionNumber = ""
    report.ReferringPhysicianName = ""
    report.Manufacturer = "Evectio"
    report.SeriesDescription = "AI DXA quality control"
    now = datetime.now().astimezone()
    report.ContentDate = now.strftime("%Y%m%d")
    report.ContentTime = now.strftime("%H%M%S.%f")
    report.CompletionFlag = "COMPLETE"
    if review.get("decision"):
        report.VerificationFlag = "VERIFIED"
        observer = Dataset()
        observer.VerifyingObserverName = str(review.get("reviewer") or "Specialist")[:64]
        observer.VerifyingOrganization = str(review.get("organization") or "")[:64]
        observer.VerificationDateTime = _dicom_datetime(str(review.get("reviewed_at", "")))
        observer.VerifyingObserverIdentificationCodeSequence = []
        report.VerifyingObserverSequence = [observer]
    else:
        report.VerificationFlag = "UNVERIFIED"
    report.ValueType = "CONTAINER"
    report.ContinuityOfContent = "SEPARATE"
    report.ConceptNameCodeSequence = [_code("DXAQC", "DXA quality control report")]

    status = "quality_issue" if result.quality_class == 1 else "acceptable"
    content = [
        _text("REGION", "Anatomical region", result.anatomical_region),
        _text(
            "PROJECTION",
            "Projection",
            f"{result.projection_code or 'UNKNOWN'} ({result.projection_source or 'unknown'})",
        ),
        _text("STATUS", "Quality status", status),
        _text("MODEL", "Model version", result.model_version),
        _text(
            "PROBABILITY",
            "Quality issue probability",
            f"{float(result.quality_prob or 0.0):.4f}",
        ),
        _text("VIOLATIONS", "Detected violations", result.violation_type or "none"),
    ]
    for index, violation in enumerate(result.violations, start=1):
        content.append(
            _text(
                f"FINDING{index}",
                "Quality finding",
                f"{violation.get('title', '')}: {violation.get('description', '')}",
            )
        )
    for key, value in sorted((result.measurements or {}).items()):
        content.append(_text("MEASURE", "Measurement", f"{key}={value}"))
    markup = result.markup or {}
    if markup:
        status = {
            "proposed": "automatic, awaiting confirmation",
            "accepted": "confirmed by specialist",
            "edited": "corrected by specialist",
            "rejected": "rejected by specialist",
        }.get(str(markup.get("status")), str(markup.get("status")))
        content.append(_text("MARKUP", "ROI markup", f"{status}; confidence {markup.get('confidence', '')}"))
        for index, check in enumerate(markup.get("checks", []), start=1):
            verdict = "ok" if check.get("ok") else "not met" if check.get("ok") is False else "unknown"
            measured = "" if check.get("measured") is None else f" {check['measured']} {check.get('unit', '')}"
            required = "" if check.get("required") is None else f" (required >= {check['required']})"
            content.append(
                _text(f"MKCHECK{index}", "Markup check", f"{check['criterion']}:{measured}{required} {verdict}")
            )
    for index, item in enumerate(result.recommendations or [], start=1):
        content.append(
            _text(
                f"RECOMMEND{index}",
                "Positioning recommendation",
                f"{item.get('criterion', '')}: {item.get('action', '')} [{item.get('measured', '')}]",
            )
        )
    if review.get("decision"):
        verdict = {
            "confirmed": "Specialist confirmed the AI conclusion",
            "rejected": "Specialist rejected the AI conclusion",
        }.get(str(review["decision"]), str(review["decision"]))
        content.append(_text("REVIEW", "Specialist review", verdict))
        content.append(
            _text("REVIEWMARKUP", "Specialist markup decision", str(review.get("markup_decision", "")))
        )
        if review.get("comment"):
            content.append(_text("REVIEWNOTE", "Specialist comment", str(review["comment"])))
    reference = Dataset()
    reference.RelationshipType = "CONTAINS"
    reference.ValueType = "IMAGE"
    reference.ConceptNameCodeSequence = [_code("SOURCE", "Source DXA image")]
    source_class_uid = safe_uid(getattr(source, "SOPClassUID", ""))
    source_instance_uid = safe_uid(getattr(source, "SOPInstanceUID", "") or result.image_uid)
    referenced_sop = Dataset()
    referenced_sop.ReferencedSOPClassUID = source_class_uid
    referenced_sop.ReferencedSOPInstanceUID = source_instance_uid
    reference.ReferencedSOPSequence = [referenced_sop]
    content.append(reference)
    report.ContentSequence = content
    # SR Document General module: the instances the report refers to.
    evidence_sop = Dataset()
    evidence_sop.ReferencedSOPClassUID = source_class_uid
    evidence_sop.ReferencedSOPInstanceUID = source_instance_uid
    evidence_series = Dataset()
    evidence_series.SeriesInstanceUID = safe_uid(getattr(source, "SeriesInstanceUID", ""))
    evidence_series.ReferencedSOPSequence = [evidence_sop]
    evidence_study = Dataset()
    evidence_study.StudyInstanceUID = report.StudyInstanceUID
    evidence_study.ReferencedSeriesSequence = [evidence_series]
    report.CurrentRequestedProcedureEvidenceSequence = [evidence_study]
    report.save_as(output, enforce_file_format=True)
    return output
