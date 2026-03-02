FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libx11-6 \
    libxcb1 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip

COPY requirements.txt .

RUN pip install --no-cache-dir \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

COPY app.py classifier.py metrics.py ./
COPY saved_models/ ./saved_models/
COPY templates/ ./templates/
COPY static/ ./static/

EXPOSE 5000

CMD ["gunicorn", "--workers", "4", "--threads", "2", "--bind", "0.0.0.0:5000", "app:app"]