# PaddleOCR 3.x plugin for Umi-OCR

This is an optional sidecar plugin. It keeps PaddleOCR's newer Python
dependencies out of Umi-OCR's bundled Python 3.8 / Qt environment and exposes
the same `code` / `data` / quadrilateral-box result shape as the existing OCR
plugins. PP-OCRv6 is the default; PP-OCRv5 remains selectable for comparison.

## Install

1. Create a separate Python environment supported by the PaddlePaddle wheel for
   your OS and CPU. Do not install PaddleOCR into Umi-OCR's embedded runtime.
2. Install the matching local inference runtime from the
   [official PaddlePaddle installation guide](https://www.paddlepaddle.org.cn/install/quick).
3. In that environment, install `paddleocr==3.7.0` from `requirements.txt`.
4. Copy this directory to `UmiOCR-data/plugins/paddleocr3`.
5. In Umi-OCR's plugin settings, set **Python executable** to the environment's
   Python executable. Select PP-OCRv6 or PP-OCRv5 in the local OCR settings.

The first engine start downloads the selected official model unless it is
already cached. For a fully offline deployment, warm the model cache while
online, then copy that cache with the environment. Runtime/model licensing and
redistribution terms must be reviewed before bundling them into an installer.

## Build a Docker image package

From the repository root, run `sh dev-tools/paddleocr3_plugin/build-image.sh`.
This builds the CPU image `umi-ocr-paddle:3.7.0` for `linux/amd64` and exports
`umi-ocr-paddle_3.7.0.tar`. Load it on another Docker host with
`docker load -i umi-ocr-paddle_3.7.0.tar`.

The image runs the same JSON-lines worker protocol used by the plugin. For a
standalone smoke run, keep stdin open and mount a persistent model cache:

```sh
docker run --rm -i -v umi-paddle-cache:/opt/paddlex \
  umi-ocr-paddle:3.7.0 --ocr-version PP-OCRv6 --cpu-threads 4
```

The first run downloads model files into the cache volume. This image packages
the OCR worker and CPU runtime; it does not contain the Umi-OCR desktop app or
GPU runtime. GPU images need a PaddlePaddle wheel and base/runtime matching the
target CUDA and driver versions. The plugin currently launches a local Python
worker, so the Docker image is a portable engine package, not yet a drop-in
Docker backend for Umi-OCR.

## Compatibility notes

- The sidecar protocol is one JSON request and one JSON response per line, so
  the model stays loaded between images.
- The adapter currently supports image OCR. Umi-OCR continues to own PDF page
  rendering, page ordering, text post-processing, and searchable-PDF writing.
- PaddleOCR 3.x results are normalized to the existing Umi-OCR plugin schema;
  coordinates are kept as four-point polygons.
- CPU/GPU runtime availability differs by platform. Choose the matching
  PaddlePaddle build using the official guide; this repository does not ship a
  model or platform-specific runtime binary.
- The smoke comparison used `enable_mkldnn=False`. In the x86 emulated test
  environment, enabling oneDNN raised a Paddle PIR runtime error. The plugin
  keeps it off by default and exposes a setting to enable it on supported
  native hardware.

## Comparison dataset

See [`../ocr-benchmark/README.md`](../ocr-benchmark/README.md) for the
reproducible synthetic corpus and the old-versus-new comparison runner.
