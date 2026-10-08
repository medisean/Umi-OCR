"""Compare the existing PaddleOCR-json pipe API with the PaddleOCR 3.x worker."""

import argparse
import base64
import json
import platform
import re
import statistics
import subprocess
import sys
import time
import unicodedata
from pathlib import Path


ROOT = Path(__file__).resolve().parent
PLUGIN_WORKER = ROOT.parent / "paddleocr3_plugin" / "worker.py"


def read_json_line(stream):
    line = stream.readline()
    if not line:
        raise RuntimeError("OCR process closed its output stream")
    return json.loads(line)


class NewEngine:
    def __init__(self, python, version, orientation, cpu_threads, mkldnn,
                 runtime_mode="python", docker_path="docker", docker_image="umi-ocr-paddle:3.7.0",
                 docker_volume="umi-ocr-paddle-cache"):
        started = time.perf_counter()
        worker_args = ["--ocr-version", version,
                       "--textline-orientation", "1" if orientation else "0",
                       "--cpu-threads", str(cpu_threads),
                       "--mkldnn", "1" if mkldnn else "0"]
        self.docker_mode = runtime_mode == "docker"
        if runtime_mode == "docker":
            command = [docker_path, "run", "--rm", "-i", "--volume",
                       "{}:/opt/paddlex".format(docker_volume), docker_image] + worker_args
        elif runtime_mode == "python":
            command = [python, str(PLUGIN_WORKER)] + worker_args
        else:
            raise ValueError("Unsupported runtime mode: {}".format(runtime_mode))
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", bufsize=1,
        )
        ready = read_json_line(self.process.stdout)
        if not ready.get("ready"):
            raise RuntimeError("PaddleOCR initialization failed: {}".format(ready.get("error")))
        self.runtime = ready.get("runtime", {})
        self.startup_seconds = time.perf_counter() - started

    def recognize(self, image_path):
        if self.docker_mode:
            with open(image_path, "rb") as image_file:
                payload = {"image_base64": base64.b64encode(image_file.read()).decode("ascii")}
        else:
            payload = {"image_path": str(image_path)}
        self.process.stdin.write(json.dumps(payload) + "\n")
        self.process.stdin.flush()
        return read_json_line(self.process.stdout)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"exit":true}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=10)


class LegacyEngine:
    def __init__(self, executable, args):
        started = time.perf_counter()
        self.process = subprocess.Popen(
            [executable] + args,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", errors="replace", bufsize=1,
        )
        # PaddleOCR-json emits a startup status line before accepting JSONL.
        while True:
            if self.process.poll() is not None:
                raise RuntimeError("PaddleOCR-json exited during initialization")
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("PaddleOCR-json closed stdout during initialization")
            if "OCR init completed." in line:
                break
        self.startup_seconds = time.perf_counter() - started

    def recognize(self, image_path):
        request = {"image_path": str(image_path)}
        self.process.stdin.write(json.dumps(request, ensure_ascii=True) + "\n")
        self.process.stdin.flush()
        return read_json_line(self.process.stdout)

    def close(self):
        if self.process.poll() is None:
            self.process.stdin.write('{"exit":""}\n')
            self.process.stdin.flush()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=5)


def response_text(response):
    if response.get("code") not in (100, 101):
        raise RuntimeError(str(response.get("data", response)))
    rows = response.get("data", [])
    if isinstance(rows, str):
        return rows
    return "\n".join(str(row.get("text", "")) for row in rows)


def normalize(text):
    return re.sub(r"\s+", "", unicodedata.normalize("NFKC", text)).casefold()


def edit_distance(left, right):
    previous = list(range(len(right) + 1))
    for i, char_left in enumerate(left, 1):
        current = [i]
        for j, char_right in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[j] + 1,
                previous[j - 1] + (char_left != char_right),
            ))
        previous = current
    return previous[-1]


def percentile(values, percent):
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def compare(engine_name, engine, samples, warmups, corpus_dir):
    if samples and warmups:
        for sample in samples[:warmups]:
            engine.recognize(corpus_dir / sample["image"])
    rows = []
    distances = 0
    characters = 0
    latencies = []
    for sample in samples:
        started = time.perf_counter()
        response = engine.recognize(corpus_dir / sample["image"])
        elapsed_ms = (time.perf_counter() - started) * 1000
        expected = sample["ground_truth"]
        actual = response_text(response)
        expected_norm, actual_norm = normalize(expected), normalize(actual)
        distance = edit_distance(expected_norm, actual_norm)
        distances += distance
        characters += len(expected_norm)
        latencies.append(elapsed_ms)
        rows.append({
            "id": sample["id"], "ground_truth": expected, "recognized": actual,
            "cer": distance / max(1, len(expected_norm)), "latency_ms": elapsed_ms,
            "exact": expected_norm == actual_norm,
        })
    return {
        "engine": engine_name,
        "startup_seconds": engine.startup_seconds,
        "exact_accuracy": sum(row["exact"] for row in rows) / max(1, len(rows)),
        "cer": distances / max(1, characters),
        "latency_median_ms": statistics.median(latencies) if latencies else 0,
        "latency_p95_ms": percentile(latencies, 0.95),
        "samples": rows,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--legacy-executable", help="PaddleOCR-json executable in pipe mode")
    parser.add_argument("--legacy-arg", action="append", default=[], help="Repeatable PaddleOCR-json startup argument")
    parser.add_argument("--new-python", default=sys.executable, help="Python executable with paddleocr 3.7.0 and its inference runtime")
    parser.add_argument("--new-runtime", choices=("python", "docker"), default="python")
    parser.add_argument("--docker-path", default="docker")
    parser.add_argument("--docker-image", default="umi-ocr-paddle:3.7.0")
    parser.add_argument("--docker-volume", default="umi-ocr-paddle-cache")
    parser.add_argument("--ocr-version", choices=("PP-OCRv5", "PP-OCRv6"), default="PP-OCRv6")
    parser.add_argument("--textline-orientation", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--mkldnn", action="store_true", help="Enable oneDNN for PaddleOCR 3.x CPU inference")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=ROOT / "corpus",
        help="Directory containing manifest.json and the referenced image files",
    )
    parser.add_argument("--dataset-name", help="Optional name recorded in the report")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-note", default="", help="Record hardware/emulation or other comparison caveats")
    args = parser.parse_args()

    corpus_dir = args.corpus_dir.expanduser().resolve()
    manifest_path = corpus_dir / "manifest.json"
    samples = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(samples, list) or not samples:
        parser.error("manifest.json must contain a non-empty JSON array of samples")
    sample_ids = set()
    for sample in samples:
        if not isinstance(sample, dict) or not all(key in sample for key in ("id", "image", "ground_truth")):
            parser.error("each manifest entry must contain id, image, and ground_truth")
        if sample["id"] in sample_ids:
            parser.error("duplicate sample id: {}".format(sample["id"]))
        sample_ids.add(sample["id"])
        image_path = (corpus_dir / sample["image"]).resolve()
        if corpus_dir not in image_path.parents or not image_path.is_file():
            parser.error("missing image or image path escapes corpus directory: {}".format(sample["image"]))
    report = {
        "dataset": args.dataset_name or corpus_dir.name,
        "sample_count": len(samples),
        "normalization": "Unicode NFKC, remove whitespace, casefold; punctuation retained",
        "system": platform.platform(),
        "architecture": platform.machine(),
        "python": platform.python_version(),
        "cpu_threads": args.cpu_threads,
        "mkldnn": args.mkldnn,
        "ocr_version": args.ocr_version,
        "legacy_command": ([args.legacy_executable] + args.legacy_arg) if args.legacy_executable else None,
        "new_runtime": None,
        "new_runtime_mode": args.new_runtime,
        "run_note": args.run_note,
        "engines": [],
    }
    engines = []
    try:
        if args.legacy_executable:
            legacy_args = list(args.legacy_arg)
            if not any(arg.startswith("-cpu_threads=") for arg in legacy_args):
                legacy_args.append("-cpu_threads={}".format(args.cpu_threads))
            engines.append(("PaddleOCR-json", LegacyEngine(args.legacy_executable, legacy_args)))
        engines.append(("PaddleOCR 3.x " + args.ocr_version,
                        NewEngine(args.new_python, args.ocr_version, args.textline_orientation,
                                  args.cpu_threads, args.mkldnn, args.new_runtime, args.docker_path,
                                  args.docker_image, args.docker_volume)))
        for name, engine in engines:
            report["engines"].append(compare(name, engine, samples, args.warmups, corpus_dir))
            if name.startswith("PaddleOCR 3.x"):
                report["new_runtime"] = engine.runtime
    finally:
        for _, engine in engines:
            engine.close()

    rendered = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
