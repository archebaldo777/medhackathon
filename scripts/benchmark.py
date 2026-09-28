#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analyzer import Analyzer  # noqa: E402
from src.batch import discover_dicom, process_directory  # noqa: E402


def percentile(values: list[float], value: float) -> float:
    return float(np.percentile(np.asarray(values), value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Воспроизводимый benchmark инференса Эвектио")
    parser.add_argument("--input", type=Path, default=PROJECT_ROOT / "dataset" / "for tests")
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "reports" / "inference_benchmark.json",
    )
    args = parser.parse_args()
    files = discover_dicom(args.input)
    if not files:
        raise SystemExit(f"DICOM не найдены: {args.input}")

    load_started = time.perf_counter()
    analyzer = Analyzer()
    model_load_ms = (time.perf_counter() - load_started) * 1000
    timings: list[float] = []
    signatures: dict[str, set[tuple]] = {str(path): set() for path in files}
    for _ in range(args.repeats):
        for path in files:
            started = time.perf_counter()
            result, _, _ = analyzer.analyze(path, path.name)
            timings.append((time.perf_counter() - started) * 1000)
            signatures[str(path)].add(
                (
                    result.processing_status,
                    result.anatomical_region,
                    result.quality_class,
                    result.violation_type,
                    result.quality_prob,
                )
            )

    with tempfile.TemporaryDirectory(prefix="dxa-benchmark-") as temporary:
        batch_started = time.perf_counter()
        batch_results = process_directory(args.input, Path(temporary) / "output", analyzer)
        batch_seconds = time.perf_counter() - batch_started
        broken = Path(temporary) / "broken.dcm"
        broken.write_bytes(b"not a dicom")
        failure, _, _ = analyzer.analyze(broken)

    payload = {
        "benchmark_version": 1,
        "model_version": analyzer.model.bundle.get("model_version"),
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "input": {
            # Relative to the project, so the report does not carry local user paths.
            "path": (
                str(args.input.resolve().relative_to(PROJECT_ROOT))
                if args.input.resolve().is_relative_to(PROJECT_ROOT)
                else args.input.name
            ),
            "dicom_files": len(files),
            "repeats": args.repeats,
            "inference_calls": len(timings),
        },
        "performance": {
            "model_load_ms": round(model_load_ms, 3),
            "per_image_ms_mean": round(statistics.mean(timings), 3),
            "per_image_ms_p50": round(percentile(timings, 50), 3),
            "per_image_ms_p95": round(percentile(timings, 95), 3),
            "per_image_ms_max": round(max(timings), 3),
            "batch_with_png_and_sr_seconds": round(batch_seconds, 3),
            "batch_images_per_second": round(len(batch_results) / max(batch_seconds, 1e-9), 3),
            "peak_rss_mb": round(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                / (1024 * 1024 if sys.platform == "darwin" else 1024),
                3,
            ),
        },
        "reliability": {
            "successful_batch_files": sum(r.processing_status == "Success" for r in batch_results),
            "failed_batch_files": sum(r.processing_status == "Failure" for r in batch_results),
            "deterministic_across_repeats": all(len(values) == 1 for values in signatures.values()),
            "malformed_dicom_controlled_failure": failure.processing_status == "Failure",
            "malformed_dicom_error": failure.error,
        },
        "notes": [
            "The benchmark uses the supplied format-check corpus and includes PNG plus DICOM SR batch output.",
            "Wall-clock values depend on host hardware; rerun on the final target environment.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
