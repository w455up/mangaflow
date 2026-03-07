FROM python:3.11-slim

# System deps for Pillow + fonts + LaMa
# libgl1-mesa-glx was renamed to libgl1 in Debian Bookworm (python:3.11-slim base)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download LaMa model
COPY download_model.py .
RUN python download_model.py && rm download_model.py

COPY . .

EXPOSE 8000

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
