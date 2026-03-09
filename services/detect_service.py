import cv2
import numpy as np
from PIL import Image
import io

def opencv_detect_bubbles(image_bytes: bytes):
    """
    Detect potential text bubbles using OpenCV.
    Returns a list of dicts: {"x", "y", "w", "h"}
    """
    # Load image
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return []

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Simple strategy: thresholding and finding contours
    # We use adaptive thresholding to handle different lighting/styles
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 11, 2
    )

    # Dilation to merge nearby text elements into boxes
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(thresh, kernel, iterations=3)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    img_h, img_w = img.shape[:2]
    
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        
        # Filter based on size (not too small, not too big)
        if w > 20 and h > 20 and w < img_w * 0.8 and h < img_h * 0.8:
            boxes.append({
                "x": float(x),
                "y": float(y),
                "w": float(w),
                "h": float(h)
            })

    # Optional: Merge overlapping boxes
    boxes = merge_boxes(boxes)

    return boxes

def merge_boxes(boxes):
    if not boxes:
        return []
    
    # Sort boxes by area descending
    boxes.sort(key=lambda b: b['w'] * b['h'], reverse=True)
    
    merged = []
    while boxes:
        curr = boxes.pop(0)
        keep = True
        for i, other in enumerate(merged):
            # If current is almost inside other, or high overlap
            if is_inside(curr, other) or overlap_ratio(curr, other) > 0.5:
                # Expand other to contain curr
                nx = min(curr['x'], other['x'])
                ny = min(curr['y'], other['y'])
                nw = max(curr['x'] + curr['w'], other['x'] + other['w']) - nx
                nh = max(curr['y'] + curr['h'], other['y'] + other['h']) - ny
                merged[i] = {"x": nx, "y": ny, "w": nw, "h": nh}
                keep = False
                break
        if keep:
            merged.append(curr)
    return merged

def is_inside(b1, b2):
    return (b1['x'] >= b2['x'] and b1['y'] >= b2['y'] and 
            b1['x'] + b1['w'] <= b2['x'] + b2['w'] and 
            b1['y'] + b1['h'] <= b2['y'] + b2['h'])

def overlap_ratio(b1, b2):
    x_left = max(b1['x'], b2['x'])
    y_top = max(b1['y'], b2['y'])
    x_right = min(b1['x'] + b1['w'], b2['x'] + b2['w'])
    y_bottom = min(b1['y'] + b1['h'], b2['y'] + b2['h'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    b1_area = b1['w'] * b1['h']
    return intersection_area / b1_area
