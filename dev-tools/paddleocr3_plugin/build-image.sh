#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
TAG=${1:-umi-ocr-paddle:3.7.0}
ARCHIVE=${2:-umi-ocr-paddle_3.7.0.tar}

docker build --platform linux/amd64 -f "$ROOT/dev-tools/paddleocr3_plugin/Dockerfile" -t "$TAG" "$ROOT"
docker save -o "$ARCHIVE" "$TAG"
printf 'Built %s and saved %s\n' "$TAG" "$ARCHIVE"
