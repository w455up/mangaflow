import httpx
from typing import List

DEEPL_API = "https://api-free.deepl.com/v2/translate"

LANG_MAP = {
    "ja": "JA", "zh-cn": "ZH", "zh-tw": "ZH",
    "ko": "KO", "en": "EN-US",
}


async def deepl_translate(
    texts: List[str],
    src_lang: str,
    dst_lang: str,
    api_key: str,
) -> List[str]:
    src = LANG_MAP.get(src_lang, src_lang.upper())
    dst = LANG_MAP.get(dst_lang, dst_lang.upper())

    payload = {
        "auth_key": api_key,
        "text": texts,
        "source_lang": src,
        "target_lang": dst,
    }
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(DEEPL_API, data=payload)
        resp.raise_for_status()
        data = resp.json()

    return [t["text"] for t in data["translations"]]
