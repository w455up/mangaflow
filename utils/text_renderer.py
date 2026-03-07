from PIL import Image, ImageDraw, ImageFont
import os
import textwrap
from typing import List, Tuple
from models import BoundingBox


FONT_PATHS = [
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJKtc-Regular.otf",
    "/System/Library/Fonts/PingFang.ttc",           # macOS
    "C:/Windows/Fonts/msjh.ttc",                    # Windows
]


def get_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # Fallback to default
    return ImageFont.load_default()


def auto_font_size(box_w: float, box_h: float, text: str, min_size=10, max_size=32) -> int:
    """Binary search for the largest font that fits the text in the box."""
    for size in range(max_size, min_size - 1, -1):
        font = get_font(size)
        # Estimate lines
        avg_char_w = size * 0.9
        chars_per_line = max(1, int(box_w / avg_char_w))
        lines = textwrap.wrap(text, width=chars_per_line) or [text]
        total_h = len(lines) * (size * 1.4)
        if total_h <= box_h:
            return size
    return min_size


def render_text_on_image(
    img: Image.Image,
    boxes: List[BoundingBox],
    font_color: str = "#000000",
    font_size_auto: bool = True,
    font_size: int = 20,
) -> Image.Image:
    """Draw translated text into each bounding box on the image."""
    result = img.copy()
    draw = ImageDraw.Draw(result)

    for b in boxes:
        if not b.trans or b.ignored:
            continue

        if font_size_auto:
            size = auto_font_size(b.w, b.h, b.trans)
        else:
            size = font_size

        font = get_font(size)
        avg_char_w = size * 0.9
        chars_per_line = max(1, int(b.w / avg_char_w))
        lines = textwrap.wrap(b.trans, width=chars_per_line) or [b.trans]
        line_h = size * 1.4

        # Vertical centering
        total_text_h = len(lines) * line_h
        start_y = b.y + max(4, (b.h - total_text_h) / 2)
        start_x = b.x + 4

        for i, line in enumerate(lines):
            y = start_y + i * line_h
            if y + size > b.y + b.h:
                break
            draw.text((start_x, y), line, font=font, fill=font_color)

    return result
