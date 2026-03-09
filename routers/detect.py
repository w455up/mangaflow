from fastapi import APIRouter, HTTPException
from models import DetectRequest, DetectResponse, BoundingBox
from services.detect_service import opencv_detect_bubbles
from services.gemini_service import gemini_detect_boxes, gemini_ocr_translate_boxes
from utils.image_utils import b64_to_pil, pil_to_b64
import io

router = APIRouter()


@router.post("", response_model=DetectResponse)
async def detect_boxes(req: DetectRequest):
    """
    Detect speech bubble bounding boxes in a manga image.
    Supports 'fast' (Gemini only) and 'precise' (OpenCV + Gemini OCR/Trans).
    """
    if not req.gemini_key:
        raise HTTPException(400, "gemini_key is required")

    try:
        img = b64_to_pil(req.image_b64, req.mime_type)
        img_w, img_h = img.width, img.height
        
        if req.mode == "precise":
            # 1. OpenCV for precise coordinates
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format=img.format or 'PNG')
            raw_boxes = opencv_detect_bubbles(img_byte_arr.getvalue())
            
            # 2. Gemini for OCR & Translation (Combined)
            # We use the raw_boxes from opencv and ask Gemini to fill in text
            mapped_results = await gemini_ocr_translate_boxes(
                image_b64=req.image_b64,
                mime_type=req.mime_type,
                img_w=img_w,
                img_h=img_h,
                boxes=raw_boxes,
                api_key=req.gemini_key,
                src_lang=req.src_lang,
                dst_lang=req.dst_lang,
                context=req.context,
                translate=req.translate
            )
            
            # 3. Final Box List
            final_boxes = []
            # Create a lookup for results
            res_map = {r['i']: r for r in mapped_results}
            for i, b in enumerate(raw_boxes):
                res = res_map.get(i, {})
                final_boxes.append(BoundingBox(
                    id=i + 1,
                    x=b['x'], y=b['y'], w=b['w'], h=b['h'],
                    orig=res.get('o', ""),
                    trans=res.get('t', ""),
                ))
            return DetectResponse(boxes=final_boxes)

        else:
            # Original Fast Mode (Gemini only)
            raw_boxes = await gemini_detect_boxes(
                image_b64=req.image_b64,
                mime_type=req.mime_type,
                img_w=img_w,
                img_h=img_h,
                api_key=req.gemini_key,
            )
            boxes = []
            for i, b in enumerate(raw_boxes):
                boxes.append(BoundingBox(
                    id=i + 1,
                    x=float(b.get("x", 0)),
                    y=float(b.get("y", 0)),
                    w=float(b.get("w", 0)),
                    h=float(b.get("h", 0)),
                    orig=b.get("text", ""),
                ))
            return DetectResponse(boxes=boxes)

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(500, f"Detection failed: {str(e)}")
