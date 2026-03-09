import json
import re
import httpx
from typing import List, Optional


GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = "gemini-2.5-flash"


async def gemini_post(api_key: str, parts: list) -> str:
    """Call Gemini generateContent and return raw text."""
    url = f"{GEMINI_API_BASE}/{GEMINI_MODEL}:generateContent?key={api_key}"
    payload = {"contents": [{"parts": parts}]}
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return text


def extract_json(raw: str):
    """Strip markdown fences and parse JSON."""
    clean = re.sub(r"```json|```", "", raw).strip()
    return json.loads(clean)


async def gemini_detect_boxes(
    image_b64: str,
    mime_type: str,
    img_w: int,
    img_h: int,
    api_key: str,
) -> List[dict]:
    """
    Ask Gemini to detect speech bubble bounding boxes.
    Returns list of {x, y, w, h, text}
    """
    prompt = (
        f"This is a manga page ({img_w}x{img_h}px). "
        "Find ALL speech bubbles and text boxes (including thought bubbles, narration boxes, SFX areas). "
        "For each one, return a JSON array with objects: "
        '{"x":<left px>,"y":<top px>,"w":<width px>,"h":<height px>,"text":"<text inside>"} '
        "Coordinates must be in the original image pixel space. "
        "Only return the JSON array, no explanation."
    )
    parts = [
        {"inline_data": {"mime_type": mime_type, "data": image_b64}},
        {"text": prompt},
    ]
    raw = await gemini_post(api_key, parts)
    return extract_json(raw)


async def gemini_ocr_boxes(
    image_b64: str,
    mime_type: str,
    img_w: int,
    img_h: int,
    boxes: List[dict],
    api_key: str,
) -> List[dict]:
    """
    Ask Gemini to read text from specific bounding boxes.
    Returns list of {i, text}
    """
    region_list = "\n".join(
        f"Box {i}: x={int(b['x'])}, y={int(b['y'])}, w={int(b['w'])}, h={int(b['h'])}"
        for i, b in enumerate(boxes)
    )
    prompt = (
        f"This is a manga image ({img_w}x{img_h}px). "
        "Please read the text inside each of the following regions precisely. "
        "Do NOT translate. Preserve original characters exactly.\n"
        f"{region_list}\n"
        'Return JSON array: [{"i":0,"text":"text in box"},...]  Only JSON, no explanation.'
    )
    parts = [
        {"inline_data": {"mime_type": mime_type, "data": image_b64}},
        {"text": prompt},
    ]
    raw = await gemini_post(api_key, parts)
    return extract_json(raw)


async def gemini_translate(
    texts: List[str],
    src_lang: str,
    dst_lang: str,
    context: Optional[str],
    api_key: str,
) -> List[str]:
    """
    Translate a list of manga dialogue strings.
    Returns list of translated strings in same order.
    """
    lang_map = {
        "ja": "Japanese", "zh-cn": "Simplified Chinese",
        "zh-tw": "Traditional Chinese", "ko": "Korean", "en": "English",
    }
    src = lang_map.get(src_lang, src_lang)
    dst = lang_map.get(dst_lang, dst_lang)
    ctx_line = f"Context: {context}\n" if context else ""

    indexed = "\n".join(f"[{i}] {t}" for i, t in enumerate(texts))
    prompt = (
        f"You are a professional manga translator. "
        f"Translate the following manga dialogue from {src} to {dst}. "
        "Keep it natural and colloquial, fitting the manga style. "
        "Preserve onomatopoeia nuance. Do NOT add explanations.\n"
        f"{ctx_line}"
        f"Source texts:\n{indexed}\n\n"
        'Return JSON array: [{"i":0,"t":"translation"},...]  Only JSON.'
    )
    raw = await gemini_post(api_key, [{"text": prompt}])
    results = extract_json(raw)
    translations = [""] * len(texts)
    for r in results:
        if 0 <= r["i"] < len(translations):
            translations[r["i"]] = r["t"]
    return translations
async def gemini_ocr_translate_boxes(
    image_b64: str,
    mime_type: str,
    img_w: int,
    img_h: int,
    boxes: List[dict],
    api_key: str,
    src_lang: str = "ja",
    dst_lang: str = "zh-tw",
    context: Optional[str] = "",
    translate: bool = True,
) -> List[dict]:
    """
    Ask Gemini to read text from specific bounding boxes AND translate them if requested.
    Returns list of {i, orig, trans}
    """
    if not boxes:
        return []

    region_list = "\n".join(
        f"Box {i}: x={int(b['x'])}, y={int(b['y'])}, w={int(b['w'])}, h={int(b['h'])}"
        for i, b in enumerate(boxes)
    )

    lang_map = {
        "ja": "Japanese", "zh-cn": "Simplified Chinese",
        "zh-tw": "Traditional Chinese", "ko": "Korean", "en": "English",
    }
    src = lang_map.get(src_lang, src_lang)
    dst = lang_map.get(dst_lang, dst_lang)
    ctx_line = f"Context: {context}\n" if context else ""

    if translate:
        prompt = (
            f"You are a professional manga translator. This is a manga image ({img_w}x{img_h}px).\n"
            f"1. Precisely read the original text from each region (preserve original characters).\n"
            f"2. Translate that text from {src} to {dst}, keeping it natural/colloquial.\n"
            f"{ctx_line}"
            f"Regions:\n{region_list}\n"
            'Return JSON array: [{"i":0,"o":"original text","t":"translation"},...]  Only JSON.'
        )
    else:
        prompt = (
            f"This is a manga image ({img_w}x{img_h}px). "
            "Please read the text inside each of the following regions precisely. "
            "Do NOT translate. Preserve original characters exactly.\n"
            f"{region_list}\n"
            'Return JSON array: [{"i":0,"o":"text in box"}]  Only JSON, no explanation.'
        )

    parts = [
        {"inline_data": {"mime_type": mime_type, "data": image_b64}},
        {"text": prompt},
    ]
    raw = await gemini_post(api_key, parts)
    results = extract_json(raw)
    
    # Return mapping
    mapped = []
    for r in results:
        idx = r.get("i")
        if idx is not None and 0 <= idx < len(boxes):
            mapped.append({
                "i": idx,
                "o": r.get("o", ""),
                "t": r.get("t", "") if translate else ""
            })
    return mapped
