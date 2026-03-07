from fastapi import APIRouter, HTTPException
from models import InpaintRequest, InpaintResponse
from services.inpaint_service import inpaint_lama
from utils.image_utils import b64_to_pil, pil_to_b64

router = APIRouter()


@router.post("", response_model=InpaintResponse)
async def run_inpaint(req: InpaintRequest):
    """
    Inpaint (erase text from) the specified bounding boxes.
    method: lama | smart | white | black
    """
    try:
        img = b64_to_pil(req.image_b64, req.mime_type)
        active_boxes = [b for b in req.boxes if not b.ignored]
        if not active_boxes:
            return InpaintResponse(image_b64=req.image_b64, mime_type="image/png")

        result = inpaint_lama(img, active_boxes, method=req.method)
        out_b64 = pil_to_b64(result, fmt="PNG")
    except Exception as e:
        raise HTTPException(500, f"Inpainting failed: {str(e)}")

    return InpaintResponse(image_b64=out_b64, mime_type="image/png")
