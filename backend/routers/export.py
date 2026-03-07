from fastapi import APIRouter, HTTPException
from fastapi.responses import Response
from models import RenderRequest, RenderResponse
from utils.image_utils import b64_to_pil, pil_to_b64
from utils.text_renderer import render_text_on_image
import base64

router = APIRouter()


@router.post("/render", response_model=RenderResponse)
async def render_translation(req: RenderRequest):
    """
    Render translated text onto the (inpainted) image.
    Returns base64 PNG.
    """
    try:
        img = b64_to_pil(req.image_b64, req.mime_type)
        result = render_text_on_image(
            img=img,
            boxes=req.boxes,
            font_color=req.font_color,
            font_size_auto=req.font_size_auto,
            font_size=req.font_size,
        )
        out_b64 = pil_to_b64(result, fmt="PNG")
    except Exception as e:
        raise HTTPException(500, f"Render failed: {str(e)}")

    return RenderResponse(image_b64=out_b64, mime_type="image/png")


@router.post("/download")
async def download_image(req: RenderRequest):
    """
    Same as /render but returns the image file directly for download.
    """
    try:
        img = b64_to_pil(req.image_b64, req.mime_type)
        result = render_text_on_image(
            img=img,
            boxes=req.boxes,
            font_color=req.font_color,
            font_size_auto=req.font_size_auto,
            font_size=req.font_size,
        )
        out_b64 = pil_to_b64(result, fmt="PNG")
        img_bytes = base64.b64decode(out_b64)
    except Exception as e:
        raise HTTPException(500, f"Export failed: {str(e)}")

    return Response(
        content=img_bytes,
        media_type="image/png",
        headers={"Content-Disposition": 'attachment; filename="translated.png"'},
    )
