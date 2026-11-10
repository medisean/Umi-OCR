# Bundled PP-OCRv6 models

This directory contains the official PP-OCRv6 medium detection and recognition
inference models used by the Docker image. PaddleX loads them from
`/opt/paddlex/official_models/` and does not need to download them at runtime.

Sources:

- [PP-OCRv6_medium_det](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_det)
- [PP-OCRv6_medium_rec](https://huggingface.co/PaddlePaddle/PP-OCRv6_medium_rec)

Both model cards declare Apache-2.0. The license text is in `LICENSE`; the
upstream model cards are preserved beside each model's inference files.
