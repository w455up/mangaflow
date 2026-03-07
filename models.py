from pydantic import BaseModel
from typing import Optional, List


class BoundingBox(BaseModel):
    id: int
    x: float   # natural image coords (px)
    y: float
    w: float
    h: float
    orig: Optional[str] = ""
    trans: Optional[str] = ""
    inpainted: bool = False
    ignored: bool = False


class DetectRequest(BaseModel):
    image_b64: str          # base64 encoded image (no data: prefix)
    mime_type: str = "image/jpeg"
    gemini_key: str


class DetectResponse(BaseModel):
    boxes: List[BoundingBox]


class OCRRequest(BaseModel):
    image_b64: str
    mime_type: str = "image/jpeg"
    gemini_key: str
    boxes: List[BoundingBox]


class OCRResponse(BaseModel):
    boxes: List[BoundingBox]


class TranslateRequest(BaseModel):
    texts: List[str]
    src_lang: str = "ja"
    dst_lang: str = "zh-tw"
    context: Optional[str] = ""
    engine: str = "gemini"       # gemini | deepl
    gemini_key: Optional[str] = None
    deepl_key: Optional[str] = None


class TranslateResponse(BaseModel):
    translations: List[str]


class InpaintRequest(BaseModel):
    image_b64: str
    mime_type: str = "image/jpeg"
    boxes: List[BoundingBox]     # boxes to inpaint (ignored=False, inpainted=False)
    method: str = "lama"         # lama | white | black | smart


class InpaintResponse(BaseModel):
    image_b64: str               # inpainted image base64
    mime_type: str = "image/png"


class RenderRequest(BaseModel):
    image_b64: str               # inpainted base64 image
    mime_type: str = "image/png"
    boxes: List[BoundingBox]
    font_color: str = "#000000"
    font_family: str = "Noto Sans TC"
    font_size_auto: bool = True
    font_size: int = 20


class RenderResponse(BaseModel):
    image_b64: str
    mime_type: str = "image/png"
