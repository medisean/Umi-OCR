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
