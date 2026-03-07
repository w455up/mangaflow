from fastapi import APIRouter, HTTPException
from models import OCRRequest, OCRResponse
from services.gemini_service import gemini_ocr_boxes
from utils.image_utils import b64_to_pil

router = APIRouter()


@router.post("", response_model=OCRResponse)
async def run_ocr(req: OCRRequest):
    """
    OCR: read text from specified bounding boxes using Gemini Vision.
    Only processes boxes that have no existing orig text.
    """
    if not req.gemini_key:
        raise HTTPException(400, "gemini_key is required")

    pending = [b for b in req.boxes if not b.orig and not b.ignored]
    if not pending:
        return OCRResponse(boxes=req.boxes)

    try:
        img = b64_to_pil(req.image_b64, req.mime_type)
        box_dicts = [{"x": b.x, "y": b.y, "w": b.w, "h": b.h} for b in pending]
        results = await gemini_ocr_boxes(
            image_b64=req.image_b64,
            mime_type=req.mime_type,
            img_w=img.width,
            img_h=img.height,
            boxes=box_dicts,
            api_key=req.gemini_key,
        )
    except Exception as e:
        raise HTTPException(500, f"OCR failed: {str(e)}")

    # Map results back
    for r in results:
        idx = r.get("i", -1)
        if 0 <= idx < len(pending):
            pending[idx].orig = r.get("text", "")

    # Merge updated pending into full list
    pending_map = {b.id: b for b in pending}
    merged = []
    for b in req.boxes:
        merged.append(pending_map.get(b.id, b))

    return OCRResponse(boxes=merged)
