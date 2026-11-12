# Small, CPU-only image for serving the classifier.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PORT=8000

WORKDIR /app

# Install dependencies first so this layer is cached between code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only what the server needs: no datasets, notebooks or training checkpoints.
COPY app.py classifier.py inference.py gunicorn.conf.py ./
COPY templates/ templates/
COPY static/ static/
COPY saved_models/dogcat_cnn.pth saved_models/gatekeeper_mobilenet_v3_small.pth saved_models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
