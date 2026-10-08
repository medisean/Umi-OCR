global_options = {
    "title": "PaddleOCR 3.x sidecar",
    "type": "group",
    "python_path": {
        "title": "Python executable",
        "default": "python",
        "toolTip": "Python executable in the isolated environment where PaddleOCR 3.x is installed.",
    },
}

local_options = {
    "title": "Text recognition (PaddleOCR 3.x)",
    "type": "group",
    "ocr_version": {
        "title": "OCR model",
        "default": "PP-OCRv6",
        "optionsList": [
            ["PP-OCRv6", "PP-OCRv6"],
            ["PP-OCRv5", "PP-OCRv5"],
        ],
    },
    "use_textline_orientation": {
        "title": "Text line orientation",
        "default": False,
        "toolTip": "Enable orientation classification for rotated text lines; this may reduce throughput.",
    },
    "cpu_threads": {
        "title": "CPU threads",
        "isInt": True,
        "default": 4,
        "min": 1,
        "max": 64,
    },
    "enable_mkldnn": {
        "title": "Enable oneDNN CPU acceleration",
        "default": False,
        "toolTip": "May improve CPU speed on supported hardware. Disable if PaddlePaddle reports a oneDNN/PIR runtime error.",
    },
}
