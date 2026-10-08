"""Generate a small, synthetic, redistributable OCR regression corpus."""

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "corpus"
CASES = [
    ("zh_simple", "离线文字识别测试", "zh-CN;clean"),
    ("en_clean", "Offline OCR benchmark 2026", "en;clean"),
    ("mixed_id", "订单号 A1024 金额 ￥58.90", "zh-CN;latin;digits"),
    ("punctuation", "括号（测试）、标点：句号。", "zh-CN;punctuation"),
    ("confusables", "0123456789 O0 I1 l1", "en;digits;confusables"),
    ("traditional", "繁體中文 OCR 測試", "zh-TW;clean"),
    ("small_text", "小字测试 123456", "zh-CN;small"),
    ("two_lines", "第一页文字\n第二行 OCR", "zh-CN;multiline"),
    ("low_contrast", "低对比度文字识别", "zh-CN;low-contrast"),
    ("rotated", "轻微倾斜 OCR 文字", "zh-CN;rotation"),
    ("blurred", "轻度模糊文本 2026", "zh-CN;blur"),
    ("dark_bg", "深色背景识别 ABC123", "zh-CN;dark-background"),
]


def find_font():
    candidates = [
        "/System/Library/Fonts/PingFang.ttc",
        "/System/Library/Fonts/STHeiti Medium.ttc",
        "/System/Library/Fonts/Hiragino Sans GB.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).is_file():
            return candidate
    raise RuntimeError("A CJK-capable font is required; install Noto Sans CJK or provide a system font.")


def main():
    random.seed(20261008)
    OUT.mkdir(parents=True, exist_ok=True)
    font_path = find_font()
    manifest = []
    for name, truth, tags in CASES:
        bg, fg, size = ("#20252b", "#f2f2f2", 45) if name == "dark_bg" else ("white", "#202020", 45)
        if name == "low_contrast":
            fg = "#b8b8b8"
        if name == "small_text":
            size = 28
        image = Image.new("RGB", (1200, 220), bg)
        draw = ImageDraw.Draw(image)
        font = ImageFont.truetype(font_path, size)
        lines = truth.split("\n")
        line_height = size + 24
        y = (image.height - len(lines) * line_height) // 2
        for line in lines:
            draw.text((50, y), line, font=font, fill=fg, stroke_width=0)
            y += line_height
        if name == "rotated":
            image = image.rotate(5, resample=Image.Resampling.BICUBIC, fillcolor="white")
        elif name == "blurred":
            image = image.filter(ImageFilter.GaussianBlur(radius=1.0))
        elif name == "low_contrast":
            image = image.resize((600, 110), Image.Resampling.BILINEAR).resize((1200, 220), Image.Resampling.BICUBIC)
        path = OUT / (name + ".png")
        image.save(path, optimize=True)
        manifest.append({"id": name, "image": path.name, "ground_truth": truth, "tags": tags.split(";")})
    (OUT / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("Generated {} samples in {}".format(len(manifest), OUT))


if __name__ == "__main__":
    main()
