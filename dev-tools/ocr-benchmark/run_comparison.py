"""Compare the existing PaddleOCR-json pipe API with the PaddleOCR 3.x worker."""

import argparse
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
    def __init__(self, python, version, orientation, cpu_threads, mkldnn):
        started = time.perf_counter()
        self.process = subprocess.Popen(
            [python, str(PLUGIN_WORKER), "--ocr-version", version,
             "--textline-orientation", "1" if orientation else "0",
             "--cpu-threads", str(cpu_threads),
             "--mkldnn", "1" if mkldnn else "0"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
            text=True, encoding="utf-8", bufsize=1,
        )
        ready = read_json_line(self.process.stdout)
        if not ready.get("ready"):
            raise RuntimeError("PaddleOCR initialization failed: {}".format(ready.get("error")))
        self.runtime = ready.get("runtime", {})
        self.startup_seconds = time.perf_counter() - started

    def recognize(self, image_path):
        self.process.stdin.write(json.dumps({"image_path": str(image_path)}) + "\n")
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


def compare(engine_name, engine, samples, warmups):
    if samples and warmups:
        for sample in samples[:warmups]:
            engine.recognize(ROOT / "corpus" / sample["image"])
    rows = []
    distances = 0
    characters = 0
    latencies = []
    for sample in samples:
        started = time.perf_counter()
        response = engine.recognize(ROOT / "corpus" / sample["image"])
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
    parser.add_argument("--ocr-version", choices=("PP-OCRv5", "PP-OCRv6"), default="PP-OCRv6")
    parser.add_argument("--textline-orientation", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--mkldnn", action="store_true", help="Enable oneDNN for PaddleOCR 3.x CPU inference")
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--run-note", default="", help="Record hardware/emulation or other comparison caveats")
    args = parser.parse_args()

    samples = json.loads((ROOT / "corpus" / "manifest.json").read_text(encoding="utf-8"))
    report = {
        "dataset": "Umi-OCR synthetic smoke corpus",
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
                        NewEngine(args.new_python, args.ocr_version, args.textline_orientation, args.cpu_threads, args.mkldnn)))
        for name, engine in engines:
            report["engines"].append(compare(name, engine, samples, args.warmups))
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
