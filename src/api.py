from __future__ import annotations

import json
import re
import shutil
import tempfile
import time
import uuid
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.background import BackgroundTask

from . import __version__
from .analyzer import Analyzer
from .batch import (
    discover_dicom,
    make_case_zip,
    make_result_zip,
    make_selection_zip,
    process_directory,
    safe_extract_zip,
)
from .constants import (
    ANATOMICAL_REGION_NAMES_RU,
    DEFAULT_PIXEL_SPACING_X_MM,
    DEFAULT_PIXEL_SPACING_Y_MM,
    HIP_INFERIOR_MARGIN_CM,
    HIP_LATERAL_MARGIN_CM,
    HIP_SUPERIOR_MARGIN_CM,
)
from .dicom_io import DICOM_SUFFIXES, is_dicom_file
from .projection import FRONTAL, LATERAL, PROJECTION_NAMES_RU
from .review import ReviewError, apply_review
from .visualization import overlay_png, source_png


MAX_UPLOAD_BYTES = 1024**3
STATIC_DIR = Path(__file__).with_name("static")
STATIC_INDEX = STATIC_DIR / "index.html"
# Only these files are served from the package directory; nothing else in it is exposed.
STATIC_ASSETS = {
    "favicon-32.png": "image/png",
    "apple-touch-icon.png": "image/png",
    "logo-mark.png": "image/png",
    "demo-spine.png": "image/png",
    "demo-heat.png": "image/png",
    "spine-3d.bin": "application/octet-stream",
}

app = FastAPI(
    title="Эвектио API",
    version=__version__,
    description="Локальная оценка качества денситометрических исследований (DXA) в формате DICOM",
)

JOB_TTL_SECONDS = 60 * 60
# Each job keeps its inputs and outputs on disk for an hour; the cap bounds
# memory and disk if many uploads arrive inside that hour.
MAX_JOBS = 32


@dataclass
class JobRecord:
    created_at: float
    work_dir: Path
    archive: Path
    results: list
    previews: dict[int, Path]

    @property
    def input_dir(self) -> Path:
        return self.work_dir / "input"

    @property
    def output_dir(self) -> Path:
        return self.work_dir / "output"


class MarkupShape(BaseModel):
    label: str = Field(max_length=128)
    points: list[list[float]] = Field(max_length=16)


class ReviewRequest(BaseModel):
    decision: str = Field(description="confirmed — специалист согласен с выводом модели, rejected — не согласен")
    reviewer: str = Field(min_length=1, max_length=64, description="ФИО или идентификатор специалиста")
    comment: str = Field(default="", max_length=1000)
    organization: str = Field(default="", max_length=64)
    markup_decision: str = Field(
        default="accepted",
        description="Решение по автоматической разметке: accepted, edited (передать markup_shapes) или rejected",
    )
    markup_shapes: list[MarkupShape] = Field(
        default_factory=list,
        max_length=16,
        description="Исправленные специалистом элементы разметки: те же подписи, точки в пикселях исходного изображения",
    )


JOBS: dict[str, JobRecord] = {}


@lru_cache(maxsize=1)
def get_analyzer() -> Analyzer:
    return Analyzer()


@app.exception_handler(Exception)
async def controlled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content={"status": "Failure", "error": f"Внутренняя ошибка обработки: {exc}"},
    )


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def index() -> HTMLResponse:
    return HTMLResponse(STATIC_INDEX.read_text(encoding="utf-8"))


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(STATIC_DIR / "favicon.ico", media_type="image/x-icon", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/static/{name}", include_in_schema=False)
def static_asset(name: str) -> FileResponse:
    media_type = STATIC_ASSETS.get(name)
    if media_type is None:
        raise HTTPException(status_code=404, detail="Файл не найден")
    return FileResponse(STATIC_DIR / name, media_type=media_type, headers={"Cache-Control": "public, max-age=86400"})


@app.get("/health")
def health() -> dict:
    analyzer = get_analyzer()
    return {
        "status": "ok",
        "service": "evectio",
        "version": __version__,
        "model_version": analyzer.model.bundle.get("model_version", "unknown"),
    }


@app.get("/v1/model")
def model_info() -> dict:
    analyzer = get_analyzer()
    bundle = analyzer.model.bundle
    return {
        "model_version": bundle.get("model_version"),
        "training_summary": bundle.get("training_summary"),
        "decision_thresholds": {
            key: {
                "threshold": round(value, 6),
                "source": analyzer.model.threshold_sources[key],
            }
            for key, value in analyzer.model.thresholds.items()
        },
        "supported_regions": ["Поясничный отдел позвоночника", "Проксимальный отдел бедра"],
        "reported_regions": list(dict.fromkeys(ANATOMICAL_REGION_NAMES_RU.values())),
        "projections": {
            "values": [PROJECTION_NAMES_RU[FRONTAL], PROJECTION_NAMES_RU[LATERAL]],
            "sources": "dicom:ViewPosition → dicom:PatientOrientation → описания серии/протокола → protocol",
        },
        "violation_types": {
            "Поясничный отдел позвоночника": [
                "Некорректная укладка",
                "Не выравнена ось позвоночника",
                "Присутствуют посторонние предметы",
            ],
            "Проксимальный отдел бедра": [
                "Некорректная укладка",
                "Некорректная область интереса",
            ],
        },
        "scanner_calibration": {
            "pixel_spacing_y_mm": DEFAULT_PIXEL_SPACING_Y_MM,
            "pixel_spacing_x_mm": DEFAULT_PIXEL_SPACING_X_MM,
            "superior_margin_cm": HIP_SUPERIOR_MARGIN_CM,
            "inferior_margin_cm": HIP_INFERIOR_MARGIN_CM,
            "lateral_margin_cm": HIP_LATERAL_MARGIN_CM,
        },
    }


async def _save_upload(upload: UploadFile, target: Path) -> None:
    written = 0
    with target.open("wb") as handle:
        while chunk := await upload.read(1024 * 1024):
            written += len(chunk)
            if written > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail=f"Файл {upload.filename} слишком большой")
            handle.write(chunk)


def _remove_expired_jobs(reserve: int = 0) -> None:
    deadline = time.time() - JOB_TTL_SECONDS
    for job_id, record in list(JOBS.items()):
        if record.created_at < deadline:
            JOBS.pop(job_id, None)
            shutil.rmtree(record.work_dir, ignore_errors=True)
    while JOBS and len(JOBS) + reserve > MAX_JOBS:
        oldest = min(JOBS, key=lambda key: JOBS[key].created_at)
        shutil.rmtree(JOBS.pop(oldest).work_dir, ignore_errors=True)


def _summary(results: list) -> dict:
    return {
        "files": len(results),
        "success": sum(item.processing_status == "Success" for item in results),
        "failure": sum(item.processing_status == "Failure" for item in results),
        "with_violations": sum(item.quality_class == 1 for item in results),
        "manual_review": sum(item.review_priority in {"high", "medium"} for item in results),
    }


def _unique(path: Path) -> Path:
    """A free path next to ``path``: name, name_2, name_3, ..."""
    if not path.exists():
        return path
    for counter in range(2, 100_000):
        candidate = path.with_name(f"{path.stem}_{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise OSError(f"Не удалось подобрать имя для {path.name}")


async def _process_uploads(files: list[UploadFile], work: Path) -> tuple[list, Path, dict[int, Path]]:
    """Store uploads so that ``path_to_study`` mirrors what the user sent.

    A single ZIP is unpacked into the input root, so paths are the paths inside
    the archive. With several uploads each ZIP gets a folder named after it and
    DICOM files keep their own names.
    """
    input_dir = work / "input"
    output_dir = work / "output"
    input_dir.mkdir()
    uploads = work / "uploads"
    uploads.mkdir()
    single_archive = len(files) == 1
    for index, upload in enumerate(files):
        original = Path(upload.filename or f"upload-{index}.dcm").name or f"upload-{index}.dcm"
        stored = uploads / f"{index:05d}"
        await _save_upload(upload, stored)
        suffix = Path(original).suffix.lower()
        if suffix == ".zip":
            target = input_dir if single_archive else _unique(input_dir / Path(original).stem)
            await run_in_threadpool(safe_extract_zip, stored, target)
            stored.unlink(missing_ok=True)
        elif suffix in DICOM_SUFFIXES or is_dicom_file(stored):
            shutil.move(stored, _unique(input_dir / original))
        else:
            raise HTTPException(
                status_code=415,
                detail=(
                    f"Неподдерживаемый тип файла: {original}. "
                    "Нужен DICOM (.dcm или файл без расширения) либо ZIP с DICOM"
                ),
            )
    if not discover_dicom(input_dir):
        raise HTTPException(status_code=400, detail="В загруженных файлах не найдено DICOM-изображений")
    # Analysis is CPU-bound: run it off the event loop so /health and the UI stay responsive.
    results = await run_in_threadpool(process_directory, input_dir, output_dir, get_analyzer(), True)
    archive = await run_in_threadpool(make_result_zip, output_dir, work / "evectio-results.zip")
    previews: dict[int, Path] = {}
    visualizations = output_dir / "visualizations"
    if visualizations.exists():
        for index in range(len(results)):
            matches = list(visualizations.glob(f"{index + 1:05d}_*.png"))
            if matches:
                previews[index] = matches[0]
    return results, archive, previews


def _job(job_id: str) -> JobRecord:
    _remove_expired_jobs()
    record = JOBS.get(job_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Результат не найден или срок его хранения истёк")
    return record


async def _run_uploads(files: list[UploadFile], work: Path) -> tuple[list, Path, dict[int, Path]]:
    """Process uploads; on any error remove the working folder and report it."""
    try:
        return await _process_uploads(files, work)
    except HTTPException:
        shutil.rmtree(work, ignore_errors=True)
        raise
    except (ValueError, OSError) as exc:
        shutil.rmtree(work, ignore_errors=True)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception:
        shutil.rmtree(work, ignore_errors=True)
        raise


@app.post(
    "/v1/analyze",
    responses={200: {"content": {"application/zip": {}}}},
    summary="Пакетно обработать DICOM-файлы или ZIP-архив",
)
async def analyze(files: list[UploadFile] = File(...)) -> FileResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Не переданы файлы")
    work = Path(tempfile.mkdtemp(prefix="evectio-api-"))
    results, result_zip, _ = await _run_uploads(files, work)
    return FileResponse(
        result_zip,
        media_type="application/zip",
        filename="evectio-results.zip",
        headers={"X-DXA-Summary": json.dumps(_summary(results), separators=(",", ":"))},
        background=BackgroundTask(shutil.rmtree, work, True),
    )


@app.post("/v1/jobs", summary="Обработать изображения и вернуть данные для веб-интерфейса")
async def create_job(files: list[UploadFile] = File(...)) -> dict:
    if not files:
        raise HTTPException(status_code=400, detail="Не переданы файлы")
    _remove_expired_jobs()
    work = Path(tempfile.mkdtemp(prefix="evectio-job-"))
    results, archive, previews = await _run_uploads(files, work)

    job_id = uuid.uuid4().hex
    _remove_expired_jobs(reserve=1)
    JOBS[job_id] = JobRecord(time.time(), work, archive, results, previews)
    payload = []
    for index, result in enumerate(results):
        item = result.api_dict()
        item["preview_url"] = f"/v1/jobs/{job_id}/preview/{index}" if index in previews else None
        item["overlay_url"] = f"/v1/jobs/{job_id}/overlay/{index}" if index in previews else None
        payload.append(item)
    return {
        "job_id": job_id,
        "expires_in_seconds": JOB_TTL_SECONDS,
        "summary": _summary(results),
        "results": payload,
        "download_url": f"/v1/jobs/{job_id}/download",
    }


@app.get("/v1/jobs/{job_id}/preview/{result_index}", include_in_schema=False)
def job_preview(job_id: str, result_index: int) -> FileResponse:
    preview = _job(job_id).previews.get(result_index)
    if preview is None or not preview.exists():
        raise HTTPException(status_code=404, detail="Визуализация не найдена")
    return FileResponse(preview, media_type="image/png")


@app.get("/v1/jobs/{job_id}/image/{result_index}", include_in_schema=False)
def job_image(job_id: str, result_index: int) -> Response:
    """Source image without overlays, for the markup editor in the web page."""
    record = _job(job_id)
    if not 0 <= result_index < len(record.results):
        raise HTTPException(status_code=404, detail="Результат не найден")
    result = record.results[result_index]
    source = (record.input_dir / result.path_to_study).resolve()
    if record.input_dir.resolve() not in source.parents or not source.exists():
        raise HTTPException(status_code=404, detail="Исходный DICOM недоступен")
    return Response(content=source_png(source), media_type="image/png")


@app.get("/v1/jobs/{job_id}/overlay/{result_index}", include_in_schema=False)
def job_overlay(job_id: str, result_index: int) -> Response:
    """AI overlay cropped to the image, the same frame as the plain image."""
    preview = _job(job_id).previews.get(result_index)
    if preview is None or not preview.exists():
        raise HTTPException(status_code=404, detail="Визуализация не найдена")
    return Response(content=overlay_png(preview), media_type="image/png")


@app.get("/v1/jobs/{job_id}/download", summary="Скачать полный архив результата")
def job_download(job_id: str) -> FileResponse:
    archive = _job(job_id).archive
    return FileResponse(
        archive,
        media_type="application/zip",
        filename="evectio-results.zip",
    )


def _safe_zip_name(stem: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "_", stem, flags=re.UNICODE).strip("_")
    return cleaned or "case"


@app.get("/v1/jobs/{job_id}/case/{result_index}/zip", summary="Скачать ZIP одного исследования из очереди")
def job_case_zip(job_id: str, result_index: int) -> FileResponse:
    """DICOM, визуализация, SR и разметка одного снимка — для просмотра вне сервиса."""
    record = _job(job_id)
    if not 0 <= result_index < len(record.results):
        raise HTTPException(status_code=404, detail="Результат не найден")
    result = record.results[result_index]
    if result.processing_status != "Success":
        raise HTTPException(status_code=409, detail="Для необработанного файла архив недоступен")
    cases_dir = record.work_dir / "cases"
    cases_dir.mkdir(exist_ok=True)
    name = f"{result_index + 1:05d}_{_safe_zip_name(Path(result.path_to_study).stem)}"
    target = cases_dir / f"{name}.zip"
    make_case_zip(record.input_dir, record.output_dir, result, target)
    return FileResponse(target, media_type="application/zip", filename=f"{name}.zip")


@app.get("/v1/jobs/{job_id}/zip", summary="Скачать ZIP отмеченных исследований (без параметра — всех)")
def job_selection_zip(job_id: str, indices: str = "") -> FileResponse:
    record = _job(job_id)
    picked: list[int] = []
    for token in indices.split(","):
        token = token.strip()
        if not token:
            continue
        if not token.isdigit() or not 0 <= int(token) < len(record.results):
            raise HTTPException(status_code=400, detail=f"Неверный номер результата: {token}")
        picked.append(int(token))
    if not picked:
        return FileResponse(record.archive, media_type="application/zip", filename="evectio-results.zip")
    picked = sorted(set(picked))
    target = record.work_dir / "evectio-selected.zip"
    make_selection_zip(record.input_dir, record.output_dir, record.results, picked, target)
    return FileResponse(target, media_type="application/zip", filename="evectio-selected.zip")


@app.post("/v1/jobs/{job_id}/review/{result_index}", summary="Подтвердить или отклонить результат специалистом")
def job_review(job_id: str, result_index: int, request: ReviewRequest) -> dict:
    """ТЗ 2.6: подтверждение или отклонение результата и предложенных корректировок.

    Решение записывается в manifest.json и в DICOM SR (VerificationFlag=VERIFIED,
    VerifyingObserverSequence); архив результата пересобирается. Колонки
    results.csv/xlsx остаются выводом модели в формате ТЗ.
    """
    record = _job(job_id)
    if not 0 <= result_index < len(record.results):
        raise HTTPException(status_code=404, detail="Результат не найден")
    result = record.results[result_index]
    source = (record.input_dir / result.path_to_study).resolve()
    if record.input_dir.resolve() not in source.parents or not source.exists():
        raise HTTPException(status_code=409, detail="Исходный DICOM недоступен для подтверждения")
    try:
        apply_review(
            record.results,
            result_index,
            record.output_dir,
            source,
            request.decision,
            request.reviewer,
            request.comment,
            request.markup_decision,
            [shape.model_dump() for shape in request.markup_shapes],
            request.organization,
        )
    except ReviewError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    record.archive.unlink(missing_ok=True)
    make_result_zip(record.output_dir, record.archive)
    return {"status": "ok", "result": result.api_dict()}
