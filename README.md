# MangaFlow 漫畫翻譯工具

## 架構

```
mangaflow/
├── backend/                  # FastAPI 後端
│   ├── main.py               # 進入點
│   ├── models.py             # Pydantic schemas
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── routers/
│   │   ├── detect.py         # POST /api/detect   — Gemini 偵測文字框
│   │   ├── ocr.py            # POST /api/ocr       — Gemini OCR
│   │   ├── translate.py      # POST /api/translate — Gemini / DeepL 翻譯
│   │   ├── inpaint.py        # POST /api/inpaint   — LaMa 消字
│   │   └── export.py         # POST /api/export/render|download
│   ├── services/
│   │   ├── gemini_service.py # Gemini Vision API 邏輯
│   │   ├── inpaint_service.py# LaMa 消字邏輯
│   │   └── deepl_service.py  # DeepL 翻譯
│   └── utils/
│       ├── image_utils.py    # PIL 工具
│       └── text_renderer.py  # 文字嵌入
└── frontend/
    └── index.html            # 前端（純 HTML/JS，呼叫後端 API）
```

## 快速啟動

### 方法一：Docker（推薦）

```bash
# 啟動所有服務
docker-compose up --build

# 前端：http://localhost:3000
# 後端：http://localhost:8000
# API 文件：http://localhost:8000/docs
```

### 方法二：手動啟動

```bash
# 安裝依賴
cd backend
pip install -r requirements.txt

# 啟動後端
uvicorn main:app --reload --port 8000

# 前端：直接用瀏覽器開啟 frontend/index.html
```

## 使用流程

1. **設定 API Key**（右側「設定」頁面）
   - 填入 Gemini API Key（必填，用於偵測、OCR、翻譯）
   - DeepL Key 選填
   - 測試後端連線

2. **上傳圖片**（支援批量）

3. **自動偵測**：呼叫 Gemini Vision 找出所有對話框

4. **辨識文字（OCR）**：Gemini 辨識每個框內文字

5. **人工調整**（可選）：
   - 拖曳白色控點調整框大小
   - 點 👁 略過不需翻譯的框
   - 雙擊框直接編輯文字

6. **翻譯**：一鍵翻譯全部，或逐框翻譯

7. **消字（LaMa）**：AI 自動填補背景，首次下載模型約 200MB

8. **預覽 / 匯出**

## API 端點

| 端點 | 說明 |
|------|------|
| `POST /api/detect` | Gemini 偵測對話框位置 |
| `POST /api/ocr` | Gemini OCR 辨識文字 |
| `POST /api/translate` | Gemini / DeepL 翻譯 |
| `POST /api/inpaint` | LaMa / 智慧填色 消字 |
| `POST /api/export/render` | 將翻譯嵌回圖片（回傳 base64） |
| `POST /api/export/download` | 下載最終圖片 |
| `GET /health` | 健康檢查 |

互動式 API 文件：http://localhost:8000/docs

## 後續 Tune 方向

- **OCR 精準度**：Prompt 工程，針對特定語言（如直排日文）調整
- **翻譯品質**：增加術語表（glossary）傳入，或換用 GPT-4o
- **消字品質**：LaMa 對純色背景效果好；複雜背景可考慮 Stability AI
- **字體**：在 Dockerfile 安裝更多中文字體，或支援上傳自訂字體
- **直排文字**：PIL text rendering 加入 vertical layout 支援
- **批量處理**：用 asyncio queue 支援多頁同時處理
