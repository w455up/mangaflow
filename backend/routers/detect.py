from fastapi import APIRouter, HTTPException
from models import DetectRequest, DetectResponse, BoundingBox
from services.gemini_service import gemini_detect_boxes
from utils.image_utils import b64_to_pil

router = APIRouter()


@router.post("", response_model=DetectResponse)
async def detect_boxes(req: DetectRequest):
    """
    Detect speech bubble bounding boxes in a manga image using Gemini Vision.
    """
    if not req.gemini_key:
        raise HTTPException(400, "gemini_key is required")

    try:
        img = b64_to_pil(req.image_b64, req.mime_type)
        raw_boxes = await gemini_detect_boxes(
            image_b64=req.image_b64,
            mime_type=req.mime_type,
            img_w=img.width,
            img_h=img.height,
            api_key=req.gemini_key,
        )
    except Exception as e:
        raise HTTPException(500, f"Detection failed: {str(e)}")

    boxes = []
    for i, b in enumerate(raw_boxes):
        try:
            boxes.append(BoundingBox(
                id=i + 1,
                x=float(b.get("x", 0)),
                y=float(b.get("y", 0)),
                w=float(b.get("w", 0)),
                h=float(b.get("h", 0)),
                orig=b.get("text", ""),
            ))
        except Exception:
            continue

    return DetectResponse(boxes=boxes)
