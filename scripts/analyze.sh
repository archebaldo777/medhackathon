#!/usr/bin/env sh
# Пакетная обработка папки с DICOM или ZIP-архива в Docker-контейнере.
#
#   ./scripts/analyze.sh INPUT [OUTPUT]
#
# INPUT  — папка с DICOM (любая вложенность) или ZIP-архив.
# OUTPUT — папка для результатов (по умолчанию ./outputs/result).
# Контейнер запускается без сети (--network none): снимки никуда не передаются.
set -eu

if [ "$#" -lt 1 ]; then
  echo "Использование: $0 INPUT [OUTPUT]" >&2
  exit 64
fi

IMAGE_NAME="${DXA_IMAGE_NAME:-evectio:1.5.0}"
INPUT="$1"
OUTPUT="${2:-outputs/result}"

if [ ! -e "$INPUT" ]; then
  echo "Не найден вход: $INPUT" >&2
  exit 66
fi

mkdir -p "$OUTPUT"
OUTPUT_ABS="$(cd "$OUTPUT" && pwd)"

if [ -d "$INPUT" ]; then
  MOUNT_DIR="$(cd "$INPUT" && pwd)"
  TARGET="/input"
else
  MOUNT_DIR="$(cd "$(dirname "$INPUT")" && pwd)"
  TARGET="/input/$(basename "$INPUT")"
fi

docker run --rm --network none \
  --user "$(id -u):$(id -g)" \
  -v "$MOUNT_DIR:/input:ro" \
  -v "$OUTPUT_ABS:/output" \
  "$IMAGE_NAME" \
  python -m src analyze "$TARGET" --output /output --zip

echo "Результаты: $OUTPUT_ABS (results.csv, results.xlsx, result.zip)"
