# First comparison run

Run date: 2026-10-08. The checked-in 12-image synthetic corpus was run through
PaddleOCR-json v1.4.1 (PP-OCRv3 model config) and PaddleOCR 3.7.0 / PP-OCRv6
medium. Both used four CPU threads. The new engine used PaddlePaddle 3.3.1 and
oneDNN disabled.

| Engine | Exact samples | Character error rate | Median latency |
| --- | ---: | ---: | ---: |
| PaddleOCR-json v1.4.1 | 10/12 (83.3%) | 2.07% | 88 ms* |
| PaddleOCR 3.7.0, PP-OCRv6 | 9/12 (75.0%) | 2.76% | 1,172 ms* |

The newer model did not win on this small synthetic set. Both engines missed
the `、` punctuation and confused `I/l` with `1`; PP-OCRv6 also converted
traditional `體` to simplified `体`. This does not establish that PP-OCRv6 is
worse on real screenshots or scanned documents. The corpus has only 12
machine-rendered examples and should be expanded with representative, labeled
images before changing the default engine.

`*` Latencies are recorded in [`results.json`](results.json), but are not a
valid performance comparison: the x86_64 Linux test ran under QEMU emulation on
an Apple ARM host, and the new engine ran with oneDNN disabled. Use native
Windows/Linux hardware and identical model/input settings for a release
performance decision.

## Bundled PP-OCRv6 Docker image smoke test

Run date: 2026-11-10. The `umi-ocr-paddle:3.7.0` amd64 image reported
PaddleOCR 3.7.0, PaddleX 3.7.2, and PaddlePaddle 3.3.1. It initialized the
bundled PP-OCRv6 medium models with Docker networking disabled and recognized
all 12 checked-in synthetic samples. Exact accuracy was 9/12 (75.0%) with
2.76% character error rate; median and p95 latency were 1,160 ms and 1,309 ms.
This ran on an arm64 Docker host under amd64 emulation, so latency is only a
smoke-test observation. The model weights are Apache-2.0 licensed and are
included in the repository and image.

The image was assembled from the already packaged runtime image and the
bundled model files because a clean runtime rebuild stalled while pip was
downloading the 195 MB PaddlePaddle wheel. The resulting image was tested
offline; a clean build from the updated Dockerfile remains to be verified on a
host with a faster PyPI connection.
