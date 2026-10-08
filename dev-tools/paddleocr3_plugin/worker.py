"""JSON-lines worker; run with the isolated PaddleOCR 3.x interpreter."""

import argparse
import base64
import contextlib
import json
import sys


def _json_line(value):
    sys.stdout.write(json.dumps(value, ensure_ascii=True, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _normalize(result):
    value = getattr(result, "json", result)
    if callable(value):
        value = value()
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict) and "res" in value:
        value = value["res"]
    if not isinstance(value, dict):
        raise ValueError("PaddleOCR returned an unsupported result object")

    texts = value.get("rec_texts", [])
    scores = value.get("rec_scores", [])
    polygons = value.get("rec_polys")
    if polygons is None or len(polygons) == 0:
        polygons = value.get("dt_polys", [])
    # PaddleOCR 3.x exposes NumPy arrays in Result.json on some releases.
    if hasattr(texts, "tolist"):
        texts = texts.tolist()
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if hasattr(polygons, "tolist"):
        polygons = polygons.tolist()
    rows = []
    for index, text in enumerate(texts):
        if not text:
            continue
        polygon = polygons[index] if index < len(polygons) else []
        points = [[int(point[0]), int(point[1])] for point in polygon]
        if len(points) < 4:
            continue
        score = float(scores[index]) if index < len(scores) else 1.0
        rows.append({"text": str(text), "box": points[:4], "score": score})
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ocr-version", choices=("PP-OCRv5", "PP-OCRv6"), required=True)
    parser.add_argument("--textline-orientation", choices=("0", "1"), default="0")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--mkldnn", choices=("0", "1"), default="0")
    args = parser.parse_args()

    try:
        # Keep model initialization logs off stdout; stdout is the protocol channel.
        with contextlib.redirect_stdout(sys.stderr):
            from paddleocr import PaddleOCR
            import paddle
            import paddleocr

            engine = PaddleOCR(
                ocr_version=args.ocr_version,
                use_doc_orientation_classify=False,
                use_doc_unwarping=False,
                use_textline_orientation=args.textline_orientation == "1",
                cpu_threads=max(1, args.cpu_threads),
                enable_mkldnn=args.mkldnn == "1",
            )
        _json_line({
            "ready": True,
            "version": args.ocr_version,
            "runtime": {
                "paddleocr": getattr(paddleocr, "__version__", "unknown"),
                "paddlepaddle": getattr(paddle, "__version__", "unknown"),
            },
        })
    except Exception as exc:
        _json_line({"ready": False, "error": "{}: {}".format(type(exc).__name__, exc)})
        return 2

    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("exit"):
                break
            if "image_path" in request:
                source = request["image_path"]
            elif "image_base64" in request:
                import cv2
                import numpy as np

                raw = base64.b64decode(request["image_base64"], validate=True)
                source = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if source is None:
                    raise ValueError("Image bytes could not be decoded")
            else:
                raise ValueError("Request must contain image_path or image_base64")

            with contextlib.redirect_stdout(sys.stderr):
                predictions = engine.predict(input=source)
            rows = []
            for prediction in predictions:
                rows.extend(_normalize(prediction))
            if rows:
                _json_line({"code": 100, "data": rows})
            else:
                _json_line({"code": 101, "data": ""})
        except Exception as exc:
            _json_line({"code": 102, "data": "[Error] {}: {}".format(type(exc).__name__, exc)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
