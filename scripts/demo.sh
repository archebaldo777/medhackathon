#!/usr/bin/env sh
set -eu

cd "$(dirname "$0")/.."
if [ -z "${DXA_PYTHON:-}" ] && [ -x ".venv/bin/python" ]; then
  PYTHON_BIN=".venv/bin/python"
else
  PYTHON_BIN="${DXA_PYTHON:-python3}"
fi
"$PYTHON_BIN" -m src analyze "dataset/for tests" --output outputs/demo --zip
echo "Demo package: outputs/demo/result.zip"

