from fastapi import APIRouter, HTTPException
from models import TranslateRequest, TranslateResponse
from services.gemini_service import gemini_translate
from services.deepl_service import deepl_translate

router = APIRouter()


@router.post("", response_model=TranslateResponse)
async def translate_texts(req: TranslateRequest):
    """
    Translate a list of strings. Engine: gemini | deepl
    """
    if not req.texts:
        return TranslateResponse(translations=[])

    if req.engine == "deepl":
        if not req.deepl_key:
            raise HTTPException(400, "deepl_key is required for DeepL engine")
        try:
            translations = await deepl_translate(
                texts=req.texts,
                src_lang=req.src_lang,
                dst_lang=req.dst_lang,
                api_key=req.deepl_key,
            )
        except Exception as e:
            raise HTTPException(500, f"DeepL translation failed: {str(e)}")

    else:  # gemini (default)
        if not req.gemini_key:
            raise HTTPException(400, "gemini_key is required for Gemini engine")
        try:
            translations = await gemini_translate(
                texts=req.texts,
                src_lang=req.src_lang,
                dst_lang=req.dst_lang,
                context=req.context,
                api_key=req.gemini_key,
            )
        except Exception as e:
            raise HTTPException(500, f"Gemini translation failed: {str(e)}")

    return TranslateResponse(translations=translations)
