# Small image for serving: ONNX Runtime + NumPy, no PyTorch.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000

WORKDIR /app

# Install dependencies first so this layer is cached between code changes.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Only what the server needs: no datasets, notebooks or training checkpoints.
COPY app.py serving.py gunicorn.conf.py ./
COPY templates/ templates/
COPY static/ static/
COPY saved_models/cnn_features.onnx saved_models/cnn_head.npz saved_models/gatekeeper.onnx saved_models/gatekeeper_classes.json saved_models/

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
CMD ["gunicorn", "-c", "gunicorn.conf.py", "app:app"]
