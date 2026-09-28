#!/usr/bin/env python3
"""Просмотр главной страницы без backend.

Отдаёт статику (index.html + 3D-модель + картинки) на стандартной библиотеке —
никаких зависимостей, pip и venv не нужны, работает на любом Python 3.

    python3 scripts/preview.py        # затем открыть http://localhost:8000

Только для просмотра вёрстки и анимации. Полноценный анализ DICOM требует
рабочего сервиса (`make serve`) с установленными numpy/scikit-learn и т. п.
"""
import http.server
import socketserver
import sys
from pathlib import Path

STATIC = Path(__file__).resolve().parent.parent / "src" / "static"
PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8000


class Handler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        path = path.split("?")[0].split("#")[0]
        if path in ("/", ""):
            path = "index.html"
        elif path.startswith("/static/"):
            path = path[len("/static/"):]
        else:
            path = path.lstrip("/")
        return str(STATIC / path)

    def log_message(self, *args):
        pass


Handler.extensions_map[".bin"] = "application/octet-stream"

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Открой http://localhost:{PORT}   (Ctrl+C — остановить)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nОстановлено.")
