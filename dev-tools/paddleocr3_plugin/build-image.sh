#!/bin/sh
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/../.." && pwd)
TAG=${1:-umi-ocr-paddle:3.7.0}
ARCHIVE=${2:-umi-ocr-paddle_3.7.0.tar}
PLATFORM=${3:-linux/amd64}
if [ "$PLATFORM" != "linux/amd64" ]; then
    echo "Unsupported platform: $PLATFORM (PaddlePaddle 3.3.1 CPU images require linux/amd64)" >&2
    exit 2
fi
CONTEXT=$(mktemp -d)
trap 'rm -rf "$CONTEXT"' EXIT HUP INT TERM

mkdir -p "$CONTEXT/dev-tools/paddleocr3_plugin/models"
cp "$ROOT/dev-tools/paddleocr3_plugin/Dockerfile" \
   "$ROOT/dev-tools/paddleocr3_plugin/.dockerignore" \
   "$ROOT/dev-tools/paddleocr3_plugin/requirements.txt" \
   "$CONTEXT/dev-tools/paddleocr3_plugin/"
cp "$ROOT/dev-tools/paddleocr3_plugin/worker.py" \
   "$CONTEXT/dev-tools/paddleocr3_plugin/"
cp "$ROOT/dev-tools/paddleocr3_plugin/models/LICENSE" \
   "$CONTEXT/dev-tools/paddleocr3_plugin/models/"
cp -R "$ROOT/dev-tools/paddleocr3_plugin/models/PP-OCRv6_medium_det" \
      "$ROOT/dev-tools/paddleocr3_plugin/models/PP-OCRv6_medium_rec" \
      "$CONTEXT/dev-tools/paddleocr3_plugin/models/"

docker build --platform "$PLATFORM" -f "$CONTEXT/dev-tools/paddleocr3_plugin/Dockerfile" -t "$TAG" "$CONTEXT"
docker save -o "$ARCHIVE" "$TAG"
printf 'Built %s and saved %s\n' "$TAG" "$ARCHIVE"
