"""Optional PaddleOCR 3.x sidecar plugin for Umi-OCR.

Copy this directory to ``UmiOCR-data/plugins/paddleocr3`` after following the
setup instructions in README.md. The PaddleOCR runtime stays isolated from the
application's bundled Python/Qt environment.
"""

from .api import Api
from .config import global_options, local_options

PluginInfo = {
    "group": "ocr",
    "global_options": global_options,
    "local_options": local_options,
    "api_class": Api,
}
