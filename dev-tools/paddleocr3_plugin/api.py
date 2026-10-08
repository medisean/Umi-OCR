"""Umi-OCR adapter that runs PaddleOCR in a separately managed Python env."""

import base64
import json
import os
import subprocess
import threading


class Api:
    def __init__(self, global_argd):
        self.python_path = global_argd.get("python_path", "python") or "python"
        self.runtime_mode = global_argd.get("runtime_mode", "python") or "python"
        self.docker_path = global_argd.get("docker_path", "docker") or "docker"
        self.docker_image = global_argd.get("docker_image", "umi-ocr-paddle:3.7.0") or "umi-ocr-paddle:3.7.0"
        self.docker_volume = global_argd.get("docker_volume", "umi-ocr-paddle-cache") or "umi-ocr-paddle-cache"
        self.worker_path = os.path.join(os.path.dirname(__file__), "worker.py")
        self.process = None
        self.lock = threading.Lock()

    def start(self, argd):
        self.stop()
        version = argd.get("ocr_version", "PP-OCRv6")
        if version not in ("PP-OCRv5", "PP-OCRv6"):
            return f"[Error] Unsupported PaddleOCR model: {version}"
        orientation = "1" if argd.get("use_textline_orientation", False) else "0"
        cpu_threads = str(max(1, int(argd.get("cpu_threads", 4))))
        mkldnn = "1" if argd.get("enable_mkldnn", False) else "0"
        try:
            worker_args = [
                "--ocr-version",
                version,
                "--textline-orientation",
                orientation,
                "--cpu-threads",
                cpu_threads,
                "--mkldnn",
                mkldnn,
            ]
            if self.runtime_mode == "docker":
                command = [
                    self.docker_path,
                    "run",
                    "--rm",
                    "-i",
                    "--volume",
                    "{}:/opt/paddlex".format(self.docker_volume),
                    self.docker_image,
                ] + worker_args
            elif self.runtime_mode == "python":
                command = [self.python_path, self.worker_path] + worker_args
            else:
                return "[Error] Unsupported runtime mode: {}".format(self.runtime_mode)
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                universal_newlines=True,
                encoding="utf-8",
                bufsize=1,
            )
            line = self.process.stdout.readline()
            message = json.loads(line) if line else {}
            if self.process.poll() is not None or not message.get("ready"):
                self.stop()
                return "[Error] PaddleOCR sidecar failed to initialize: {}".format(
                    message.get("error", "worker exited before reporting readiness")
                )
            return ""
        except Exception as exc:
            self.stop()
            return "[Error] Unable to start PaddleOCR sidecar: {}".format(exc)

    def stop(self):
        process = self.process
        self.process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin:
                process.stdin.write('{"exit":true}\n')
                process.stdin.flush()
                process.wait(timeout=5)
        except Exception:
            try:
                process.terminate()
                process.wait(timeout=3)
            except Exception:
                process.kill()
        finally:
            for stream in (process.stdin, process.stdout):
                if stream:
                    try:
                        stream.close()
                    except Exception:
                        pass

    def runPath(self, img_path):
        if self.runtime_mode == "docker":
            try:
                with open(img_path, "rb") as image_file:
                    return self.runBytes(image_file.read())
            except Exception as exc:
                return {"code": 102, "data": "[Error] Unable to read image: {}".format(exc)}
        return self._request({"image_path": os.path.abspath(img_path)})

    def runBytes(self, image_bytes):
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return self._request({"image_base64": encoded})

    def runBase64(self, image_base64):
        prefix = "base64,"
        if image_base64.startswith("data:") and prefix in image_base64:
            image_base64 = image_base64.split(prefix, 1)[1]
        return self._request({"image_base64": image_base64})

    def _request(self, payload):
        with self.lock:
            process = self.process
            if process is None or process.poll() is not None:
                return {"code": 102, "data": "[Error] PaddleOCR sidecar is not running."}
            try:
                process.stdin.write(json.dumps(payload, ensure_ascii=True) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
                response = json.loads(line) if line else {}
                if "code" in response:
                    return response
                return {
                    "code": 102,
                    "data": "[Error] Invalid PaddleOCR sidecar response: {}".format(
                        line[:300]
                    ),
                }
            except Exception as exc:
                return {"code": 102, "data": "[Error] PaddleOCR request failed: {}".format(exc)}
