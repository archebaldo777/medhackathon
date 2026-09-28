from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyzer import Analyzer
from .batch import make_result_zip, process_archive, process_directory


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(
        prog="evectio",
        description="Контроль качества денситометрических исследований (DXA DICOM)",
    )
    subcommands = root.add_subparsers(dest="command", required=True)
    analyze = subcommands.add_parser(
        "analyze", help="Обработать DICOM, каталог или ZIP"
    )
    analyze.add_argument(
        "input", type=Path, help="DICOM-файл, папка с DICOM (любая вложенность) или ZIP-архив"
    )
    analyze.add_argument(
        "--output", type=Path, default=Path("outputs"), help="папка для результатов (по умолчанию outputs)"
    )
    analyze.add_argument("--model", type=Path, default=None, help="путь к файлу модели .joblib")
    analyze.add_argument(
        "--no-visualizations", action="store_true", help="не строить PNG и DICOM-визуализации (быстрее)"
    )
    analyze.add_argument("--zip", action="store_true", help="Создать result.zip")
    serve = subcommands.add_parser("serve", help="Запустить API и веб-интерфейс")
    serve.add_argument("--host", default="0.0.0.0", help="адрес (по умолчанию 0.0.0.0)")
    serve.add_argument("--port", type=int, default=8000, help="порт (по умолчанию 8000)")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "serve":
        import uvicorn

        uvicorn.run("src.api:app", host=args.host, port=args.port)
        return

    try:
        analyzer = Analyzer(args.model) if args.model else Analyzer()
        visualizations = not args.no_visualizations
        if args.input.suffix.lower() == ".zip":
            results = process_archive(args.input, args.output, analyzer, visualizations)
        else:
            results = process_directory(args.input, args.output, analyzer, visualizations)
        if args.zip:
            make_result_zip(args.output, args.output / "result.zip")
    except Exception as exc:
        print(
            json.dumps(
                {"status": "Failure", "input": str(args.input), "error": str(exc)},
                ensure_ascii=False,
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    summary = {
        "files": len(results),
        "success": sum(r.processing_status == "Success" for r in results),
        "failure": sum(r.processing_status == "Failure" for r in results),
        "with_violations": sum(r.quality_class == 1 for r in results),
        "output": str(args.output.resolve()),
    }
    print(json.dumps(summary, ensure_ascii=False))
    raise SystemExit(0 if summary["failure"] == 0 else 2)


if __name__ == "__main__":
    main()
