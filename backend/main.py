from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from routers import detect, ocr, translate, inpaint, export

app = FastAPI(title="MangaFlow API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(detect.router,    prefix="/api/detect",    tags=["detect"])
app.include_router(ocr.router,       prefix="/api/ocr",       tags=["ocr"])
app.include_router(translate.router, prefix="/api/translate", tags=["translate"])
app.include_router(inpaint.router,   prefix="/api/inpaint",   tags=["inpaint"])
app.include_router(export.router,    prefix="/api/export",    tags=["export"])

# Serve frontend
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))

@app.get("/health")
def health():
    return {"status": "ok"}
