from __future__ import annotations

import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

from pydicom.uid import generate_uid

from .analyzer import AnalysisResult, Analyzer
from .dicom_gsps import save_markup_gsps
from .dicom_io import is_dicom_file
from .dicom_sr import safe_uid, save_dicom_sr
from .dicom_viz import save_dicom_visualization
from .reporting import write_csv, write_manifest, write_xlsx
from .visualization import save_visualization


MAX_ARCHIVE_FILES = 10_000
MAX_UNCOMPRESSED_BYTES = 5 * 1024**3


class UnsafeArchiveError(ValueError):
    pass


ZIP_UTF8_FLAG = 0x800


def zip_member_name(item: zipfile.ZipInfo) -> str:
    """Member name decoded the way the archiver meant it.

    Without the UTF-8 flag ``zipfile`` decodes names as cp437. Command-line zip
    on Linux/macOS stores UTF-8 bytes without the flag, and Windows'
    built-in compression stores Cyrillic names in cp866; both are recovered.
    """
    if item.flag_bits & ZIP_UTF8_FLAG:
        return item.filename
    try:
        raw = item.filename.encode("cp437")
    except UnicodeEncodeError:
        return item.filename
    for encoding in ("utf-8", "cp866"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return item.filename


def safe_extract_zip(archive: str | Path, destination: str | Path) -> list[Path]:
    target = Path(destination).resolve()
    extracted: list[Path] = []
    try:
        bundle = zipfile.ZipFile(archive)
    except (zipfile.BadZipFile, zipfile.LargeZipFile) as exc:
        raise UnsafeArchiveError(f"Файл не является корректным ZIP-архивом: {Path(archive).name}") from exc
    with bundle:
        members = bundle.infolist()
        if len(members) > MAX_ARCHIVE_FILES:
            raise UnsafeArchiveError("Слишком много файлов в архиве")
        if sum(item.file_size for item in members) > MAX_UNCOMPRESSED_BYTES:
            raise UnsafeArchiveError("Распакованный архив превышает допустимый размер")
        for item in members:
            candidate = (target / zip_member_name(item)).resolve()
            if candidate != target and target not in candidate.parents:
                raise UnsafeArchiveError("Архив содержит небезопасный путь")
            if item.is_dir():
                candidate.mkdir(parents=True, exist_ok=True)
                continue
            candidate.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(item) as source, candidate.open("wb") as sink:
                shutil.copyfileobj(source, sink)
            extracted.append(candidate)
    return extracted


def discover_dicom(path: str | Path) -> list[Path]:
    """DICOM files in a file or folder tree: ``.dcm``/``.dicom`` or any file with the DICM signature."""
    source = Path(path)
    if source.is_file():
        return [source] if is_dicom_file(source) else []
    if source.is_dir():
        return sorted(
            item
            for item in source.rglob("*")
            if not any(part.startswith(".") or part == "__MACOSX" for part in item.relative_to(source).parts)
            and is_dicom_file(item)
        )
    return []


def process_directory(
    input_path: str | Path,
    output_dir: str | Path,
    analyzer: Analyzer | None = None,
    visualizations: bool = True,
) -> list[AnalysisResult]:
    source = Path(input_path).resolve()
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    engine = analyzer or Analyzer()
    files = discover_dicom(source)
    if not files:
        raise ValueError(
            f"DICOM-файлы не найдены (нужны .dcm, файлы DICOM без расширения или ZIP с ними): {source}"
        )

    analysis_items = []
    for path in files:
        try:
            relative = str(path.relative_to(source)) if source.is_dir() else path.name
        except ValueError:
            relative = path.name
        analysis_items.append((path, relative))

    analyzed = engine.analyze_many(analysis_items)
    results: list[AnalysisResult] = []
    # One derived visualisation series per source study, as a viewer expects.
    viz_series: dict[str, tuple[str, int]] = {}
    markup_series: dict[str, str] = {}
    for index, (path, analyzed_item) in enumerate(zip(files, analyzed), start=1):
        result, normalized, prediction = analyzed_item
        results.append(result)
        artifacts_started = time.perf_counter()
        if result.processing_status == "Success":
            sr_name = f"{index:05d}_{path.stem}_sr.dcm"
            try:
                save_dicom_sr(path, result, output / "dicom_sr" / sr_name)
                result.artifacts["dicom_sr"] = f"dicom_sr/{sr_name}"
            except Exception as exc:
                result.error = f"DICOM SR не создан: {exc}"
            if result.markup.get("shapes"):
                markup_name = f"{index:05d}_{path.stem}_markup.dcm"
                study_key = safe_uid(result.study_uid)
                try:
                    save_markup_gsps(
                        path,
                        result.markup,
                        output / "dicom_markup" / markup_name,
                        series_uid=markup_series.setdefault(study_key, generate_uid()),
                        instance_number=index,
                    )
                    result.artifacts["dicom_markup"] = f"dicom_markup/{markup_name}"
                except Exception as exc:
                    result.error = f"Разметка в DICOM не сохранена: {exc}"
        if visualizations and normalized is not None and prediction is not None:
            name = f"{index:05d}_{path.stem}.png"
            try:
                save_visualization(
                    engine.model,
                    normalized,
                    prediction,
                    output / "visualizations" / name,
                    pixel_spacing_y_mm=result.pixel_spacing_y_mm,
                    pixel_spacing_x_mm=result.pixel_spacing_x_mm,
                    markup=result.markup,
                )
                result.artifacts["visualization"] = f"visualizations/{name}"
            except Exception as exc:
                result.error = f"Визуализация не создана: {exc}"
            if "visualization" in result.artifacts and result.processing_status == "Success":
                study_key = safe_uid(result.study_uid)
                series_uid, count = viz_series.get(study_key, (generate_uid(), 0))
                viz_series[study_key] = (series_uid, count + 1)
                viz_name = f"{index:05d}_{path.stem}_viz.dcm"
                try:
                    save_dicom_visualization(
                        path,
                        result,
                        output / result.artifacts["visualization"],
                        output / "dicom_viz" / viz_name,
                        series_uid=series_uid,
                        instance_number=count + 1,
                    )
                    result.artifacts["dicom_visualization"] = f"dicom_viz/{viz_name}"
                except Exception as exc:
                    result.error = f"DICOM-визуализация не создана: {exc}"
        # time_of_processing covers the image's own artefacts as well.
        result.time_of_processing = round(
            result.time_of_processing + time.perf_counter() - artifacts_started, 4
        )

    write_csv(results, output / "results.csv")
    write_xlsx(results, output / "results.xlsx")
    write_manifest(results, output / "manifest.json")
    return results


def make_result_zip(output_dir: str | Path, destination: str | Path) -> Path:
    output = Path(output_dir)
    target = Path(destination)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(output.rglob("*")):
            if path.is_file() and path.resolve() != target.resolve():
                bundle.write(path, path.relative_to(output))
    return target


def case_files(input_dir: str | Path, output_dir: str | Path, result: AnalysisResult) -> list[tuple[Path, str]]:
    """(absolute path, archive name) pairs for one study: its source DICOM and artefacts.

    Used both for a single-case download and to assemble a selection of cases, so a
    reviewer working through the worklist can pull one study — or a marked subset —
    without the rest of the batch.
    """
    input_root = Path(input_dir).resolve()
    output_root = Path(output_dir)
    items: list[tuple[Path, str]] = []
    source = (input_root / result.path_to_study).resolve()
    if input_root in source.parents and source.exists():
        items.append((source, f"dicom/{result.path_to_study}"))
    for relative in result.artifacts.values():
        path = output_root / relative
        if path.exists():
            items.append((path, relative))
    return items


def make_case_zip(
    input_dir: str | Path, output_dir: str | Path, result: AnalysisResult, destination: str | Path
) -> Path:
    """ZIP of one study: source DICOM, its visualisation, SR and markup, and a case summary."""
    target = Path(destination)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path, arcname in case_files(input_dir, output_dir, result):
            bundle.write(path, arcname)
        bundle.writestr("case.json", json.dumps(result.api_dict(), ensure_ascii=False, indent=2, default=str))
    return target


def make_selection_zip(
    input_dir: str | Path,
    output_dir: str | Path,
    results: list[AnalysisResult],
    indices: list[int],
    destination: str | Path,
) -> Path:
    """ZIP of several studies, each kept in its own numbered folder."""
    target = Path(destination)
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for index in indices:
            result = results[index]
            folder = f"{index + 1:05d}_{Path(result.path_to_study).stem}"
            for path, arcname in case_files(input_dir, output_dir, result):
                bundle.write(path, f"{folder}/{arcname}")
            bundle.writestr(
                f"{folder}/case.json", json.dumps(result.api_dict(), ensure_ascii=False, indent=2, default=str)
            )
    return target


def process_archive(
    archive: str | Path,
    output_dir: str | Path,
    analyzer: Analyzer | None = None,
    visualizations: bool = True,
) -> list[AnalysisResult]:
    with tempfile.TemporaryDirectory(prefix="evectio-input-") as temporary:
        safe_extract_zip(archive, temporary)
        return process_directory(temporary, output_dir, analyzer, visualizations)
