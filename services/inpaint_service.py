"""
LaMa inpainting service.
Uses simple-lama-inpainting (pip install simple-lama-inpainting).
Falls back to smart color-fill if LaMa model is not available.
"""
import io
import numpy as np
from PIL import Image, ImageFilter
from typing import List
import cv2
from models import BoundingBox


_lama = None
_lama_loaded = False


def _get_lama():
    global _lama, _lama_loaded
    if _lama_loaded:
        return _lama
    try:
        from simple_lama_inpainting import SimpleLama
        _lama = SimpleLama()
        print("[LaMa] Model loaded successfully.")
    except Exception as e:
        print(f"[LaMa] Could not load model: {e}  → falling back to smart-fill.")
        _lama = None
    _lama_loaded = True
    return _lama


def _smart_fill_color(img: Image.Image, x: int, y: int, w: int, h: int) -> tuple:
    """Sample border pixels around the box to estimate background color."""
    arr = np.array(img)
    H, W = arr.shape[:2]
    samples = []
    pad = 6

    # Top border
    r1, r2 = max(0, y - pad), max(0, y)
    if r2 > r1:
        samples.append(arr[r1:r2, max(0,x):min(W,x+w)])
    # Bottom border
    r1, r2 = min(H, y+h), min(H, y+h+pad)
    if r2 > r1:
        samples.append(arr[r1:r2, max(0,x):min(W,x+w)])
    # Left border
    c1, c2 = max(0, x-pad), max(0, x)
    if c2 > c1:
        samples.append(arr[max(0,y):min(H,y+h), c1:c2])
    # Right border
    c1, c2 = min(W, x+w), min(W, x+w+pad)
    if c2 > c1:
        samples.append(arr[max(0,y):min(H,y+h), c1:c2])

    if samples:
        all_pixels = np.concatenate([s.reshape(-1, s.shape[-1]) for s in samples if s.size > 0])
        color = tuple(int(v) for v in np.median(all_pixels, axis=0)[:3])
        return color
    return (255, 255, 255)


def inpaint_lama(
    img: Image.Image,
    boxes: List[BoundingBox],
    method: str = "lama",
) -> Image.Image:
    """
    Inpaint text regions.
    method: 'lama' | 'opencv' | 'smart' | 'white' | 'black'
    """
    result = img.copy().convert("RGB")

    if method == "lama":
        lama = _get_lama()
        if lama is not None:
            return _inpaint_with_lama(lama, result, boxes)
        # fallback
        print("[LaMa] Using opencv fallback.")
        method = "opencv"

    if method == "opencv":
        return _inpaint_with_opencv(result, boxes)

    # Pixel-fill methods
    from PIL import ImageDraw
    draw = ImageDraw.Draw(result)
    for b in boxes:
        if b.ignored:
            continue
        x, y, w, h = int(b.x), int(b.y), int(b.w), int(b.h)
        if method == "white":
            fill = (255, 255, 255)
        elif method == "black":
            fill = (0, 0, 0)
        else:  # smart
            fill = _smart_fill_color(result, x, y, w, h)
        draw.rectangle([x, y, x + w, y + h], fill=fill)

    return result


def _inpaint_with_lama(lama, img: Image.Image, boxes: List[BoundingBox]) -> Image.Image:
    """
    Use LaMa model to inpaint each box individually, then composite back.
    We build a single mask covering all boxes and run one pass.
    """
    mask = Image.new("L", img.size, 0)
    from PIL import ImageDraw
    mask_draw = ImageDraw.Draw(mask)
    has_box = False
    for b in boxes:
        if b.ignored:
            continue
        # Slightly expand mask for cleaner fill
        pad = 4
        x1 = max(0, int(b.x) - pad)
        y1 = max(0, int(b.y) - pad)
        x2 = min(img.width,  int(b.x + b.w) + pad)
        y2 = min(img.height, int(b.y + b.h) + pad)
        mask_draw.rectangle([x1, y1, x2, y2], fill=255)
        has_box = True

    if not has_box:
        return img

    try:
        result = lama(img, mask)
        return result
    except Exception as e:
        print(f"[LaMa] Inference error: {e}  → smart-fill fallback.")
        # fallback
        from PIL import ImageDraw as ID
        r = img.copy()
        d = ID.Draw(r)
        for b in boxes:
            if b.ignored:
                continue
            x, y, w, h = int(b.x), int(b.y), int(b.w), int(b.h)
            fill = _smart_fill_color(r, x, y, w, h)
            d.rectangle([x, y, x+w, y+h], fill=fill)
        return r


def _inpaint_with_opencv(img: Image.Image, boxes: List[BoundingBox]) -> Image.Image:
    """
    Use OpenCV traditional inpainting with bubble color detection.
    For speech bubbles, we fill text with the background color for 100% sharpness.
    For textures, we use Navier-Stokes inpainting.
    """
    # Convert PIL to OpenCV (BGR)
    open_cv_image = np.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()

    # Create master masks
    # full_mask: for cv2.inpaint
    # fill_mask: for direct color fill (sharpest)
    full_mask = np.zeros(open_cv_image.shape[:2], dtype=np.uint8)
    
    for b in boxes:
        if b.ignored: continue
        x1, y1 = max(0, int(b.x)), max(0, int(b.y))
        x2, y2 = min(img.width, int(b.x + b.w)), min(img.height, int(b.y + b.h))
        if x2 <= x1 or y2 <= y1: continue
        
        roi = open_cv_image[y1:y2, x1:x2]
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 1. Precise Text Detection
        thresh = cv2.adaptiveThreshold(
            gray_roi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 15, 10
        )
        
        # Clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.dilate(thresh, kernel, iterations=1)

        # 2. Bubble Detection (is this area mostly a solid color?)
        # Get dominant color or average of edges
        edge_pixels = np.concatenate([roi[0,:], roi[-1,:], roi[:,0], roi[:,-1]])
        avg_color = np.mean(edge_pixels, axis=0) # BGR
        
        # If edges are very similar (low std dev) and mostly bright, it's a bubble
        std_color = np.std(edge_pixels, axis=0)
        is_bubble = np.max(std_color) < 15 and np.mean(avg_color) > 200 # Likely white/light bubble
        
        if is_bubble:
            # For Bubbles: Fill the text strokes directly with the bubble color
            # This is much sharper than inpainting
            roi_mask = (thresh > 0)
            roi[roi_mask] = avg_color
            open_cv_image[y1:y2, x1:x2] = roi
        else:
            # For Contextual Backgrounds: Mark for cv2.inpaint
            full_mask[y1:y2, x1:x2] = cv2.bitwise_or(full_mask[y1:y2, x1:x2], thresh)

    # Run inpaint only for non-bubble areas
    if np.any(full_mask > 0):
        dst = cv2.inpaint(open_cv_image, full_mask, 3, cv2.INPAINT_NS)
    else:
        dst = open_cv_image

    # Convert BGR back to RGB PIL
    dst = cv2.cvtColor(dst, cv2.COLOR_BGR2RGB)
    return Image.fromarray(dst)
