#!/usr/bin/env sh
set -eu

IMAGE_NAME="${DXA_IMAGE_NAME:-evectio:1.5.0}"
docker build --pull -t "$IMAGE_NAME" .
echo "Built $IMAGE_NAME"
