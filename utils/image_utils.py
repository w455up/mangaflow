import base64
import io
from PIL import Image


def b64_to_pil(b64: str, mime_type: str = "image/jpeg") -> Image.Image:
    """Decode base64 string to PIL Image."""
    data = base64.b64decode(b64)
    return Image.open(io.BytesIO(data)).convert("RGB")


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    """Encode PIL Image to base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def crop_box(img: Image.Image, x: float, y: float, w: float, h: float) -> Image.Image:
    """Crop a region from PIL Image using natural coords."""
    return img.crop((int(x), int(y), int(x + w), int(y + h)))
