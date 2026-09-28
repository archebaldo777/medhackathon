FROM python:3.12.10-slim-bookworm@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN groupadd --system evectio && useradd --system --gid evectio --home-dir /app evectio
COPY requirements.txt ./
RUN pip install --disable-pip-version-check -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
# Only the model the service loads; older bundles stay out of the image.
COPY models/dxa_quality_v4.joblib ./models/
# Code and model must be readable by any UID: scripts/analyze.sh runs the container
# as the host user so that results in the mounted folder belong to that user.
RUN pip install --no-deps . \
    && chown -R evectio:evectio /app \
    && chmod -R a+rX /app

USER evectio
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

