#!/usr/bin/env sh
set -eu

IMAGE_NAME="${DXA_IMAGE_NAME:-evectio:1.5.0}"
SERVICE_PORT="${DXA_PORT:-8000}"
docker run --rm --name evectio -p "$SERVICE_PORT:8000" "$IMAGE_NAME"
