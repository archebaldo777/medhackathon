#!/bin/sh
cd "$(dirname "$0")"
echo "=== Эвектио: запуск локального веб-сервера ==="
if [ -x ".venv/bin/python" ]; then
  exec .venv/bin/python -m src serve
else
  echo "No .venv found, falling back to system python3 (make sure deps are installed)"
  exec python3 -m src serve
fi
