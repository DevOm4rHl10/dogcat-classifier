"""
Meowf Investigations: check that a photo shows a dog or a cat, classify it, and
explain the decision with Grad-CAM (drawn as red-ink "evidence"). Test-set results are shown at /metrics.

Run locally:   flask --app app run --debug
Production:    gunicorn -c gunicorn.conf.py app:app
"""
import base64
import json
import os
import random
from io import BytesIO

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, render_template, request
from PIL import Image, ImageOps, UnidentifiedImageError

from classifier import DogCatCNN
from inference import CLASSES, INPUT_SIZE, Gatekeeper, cnn_tensor, load_image

# ---------------------------------------------------------------------------
# Configuration (override with environment variables)
# ---------------------------------------------------------------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "saved_models/dogcat_cnn.pth")
GATE_PATH = os.environ.get("GATE_PATH", "saved_models/gatekeeper_mobilenet_v3_small.pth")
GATE_THRESHOLD = float(os.environ.get("GATE_THRESHOLD", "0.2"))
DISPLAY_SIZE = 384   # size of the Grad-CAM image sent to the browser

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # reject uploads over 10 MB
METRICS_JSON = os.path.join(app.root_path, "static", "metrics", "metrics.json")

# ---------------------------------------------------------------------------
# Models: loaded once, when the module is imported (gunicorn preloads it)
# ---------------------------------------------------------------------------
torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "2")))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DogCatCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()
for p in model.parameters():          # Grad-CAM only needs gradients w.r.t. activations
    p.requires_grad_(False)

if os.path.exists(GATE_PATH):
    gatekeeper = Gatekeeper(GATE_PATH, device, GATE_THRESHOLD)
else:
    gatekeeper = None
    app.logger.warning("Gatekeeper weights not found at %s: every photo goes straight to the CNN. "
                       "Run scripts/download_gatekeeper.py to enable it.", GATE_PATH)


# ---------------------------------------------------------------------------
# Prediction + Grad-CAM in a single forward/backward pass
# ---------------------------------------------------------------------------
def predict_with_gradcam(x: torch.Tensor):
    """Return (label, confidence, cam) where cam is an INPUT_SIZE x INPUT_SIZE array in [0, 1]."""
    with torch.no_grad():
        feats = model.features(x)          # 1 x C x h x w, no graph needed for the conv layers
    # Local leaf tensor: no hooks on the shared model, so concurrent requests can't mix gradients.
    feats = feats.detach().requires_grad_(True)
    with torch.enable_grad():
        logit = model.classify(feats)      # 1 x 1, raw score: > 0 means dog

        p_dog = torch.sigmoid(logit).item()
        is_dog = p_dog > 0.5

        # Explain the class that was predicted: +logit for dog, -logit for cat.
        # Using the logit (not the sigmoid) avoids vanishing gradients on confident images.
        score = logit if is_dog else -logit
        score.sum().backward()

    grads = feats.grad[0]                  # C x h x w
    acts = feats.detach()[0]
    weights = grads.mean(dim=(1, 2))       # global-average-pooled gradients
    cam = F.relu((weights[:, None, None] * acts).sum(dim=0))
    cam = F.interpolate(cam[None, None], size=(INPUT_SIZE, INPUT_SIZE),
                        mode="bilinear", align_corners=False)[0, 0]
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    label = CLASSES[1] if is_dog else CLASSES[0]
    confidence = p_dog if is_dog else 1.0 - p_dog
    return label, confidence, cam.cpu().numpy()


# Evidence print: the photo as a sepia print, with the strongest Grad-CAM evidence
# marked in red ink (the site's detective theme). Weak evidence is left unmarked.
PAPER = np.array([0.97, 0.94, 0.87], dtype=np.float32)
INK = np.array([0.16, 0.13, 0.11], dtype=np.float32)
RED_INK = np.array([0.70, 0.15, 0.12], dtype=np.float32)


def overlay_b64(img: Image.Image, cam: np.ndarray) -> str:
    """Return a base64 JPEG of the (squashed, like the model input) photo with red-ink evidence."""
    size = (DISPLAY_SIZE, DISPLAY_SIZE)
    gray = ImageOps.autocontrast(img.convert("L").resize(size, Image.BILINEAR), cutoff=1)
    photo = np.asarray(gray, dtype=np.float32)[..., None] / 255.0
    base = INK + (PAPER - INK) * photo
    cam_big = np.asarray(Image.fromarray(np.uint8(cam * 255)).resize(size, Image.BILINEAR),
                         dtype=np.float32) / 255.0
    ink = (np.clip((cam_big - 0.3) / 0.7, 0, 1) ** 0.8 * 0.9)[..., None]
    out = base * (1 - ink) + base * RED_INK * ink          # multiply blend, weighted by evidence
    buf = BytesIO()
    Image.fromarray(np.uint8(np.clip(out, 0, 1) * 255)).save(buf, format="JPEG", quality=85)
    return base64.b64encode(buf.getvalue()).decode()


def readable(imagenet_label: str) -> str:
    """'ashcan, trash can, garbage can' -> 'ashcan'; keeps the answer short."""
    return imagenet_label.split(",")[0].strip()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.context_processor
def case_number():
    """Every page opens a fresh case file."""
    return {"case_no": random.randint(1000, 9999)}


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/predict")
def predict():
    file = request.files.get("file")
    if file is None or file.filename == "":
        return render_template("index.html", error="Choose a photo to classify."), 400

    try:
        img = load_image(file.stream)
    except (UnidentifiedImageError, OSError):
        return render_template(
            "index.html", error="That file couldn't be read as an image. Upload a JPG, PNG or WebP."
        ), 400

    forced = request.form.get("force") == "1"
    if gatekeeper is not None and not forced:
        is_pet, pet_prob, top_label = gatekeeper.check(img)
        if not is_pet:
            return render_template("index.html", rejected=True,
                                   looks_like=readable(top_label), pet_prob=pet_prob)

    label, confidence, cam = predict_with_gradcam(cnn_tensor(img).unsqueeze(0).to(device))
    return render_template(
        "index.html",
        prediction=label,
        confidence=confidence,
        gradcam_image=overlay_b64(img, cam),
        forced=forced,
    )


@app.get("/metrics")
def metrics():
    try:
        with open(METRICS_JSON) as f:
            report = json.load(f)
    except FileNotFoundError:
        report = None
    return render_template("metrics.html", r=report)


@app.errorhandler(413)
def too_large(_):
    return render_template("index.html", error="That photo is larger than 10 MB. Choose a smaller one."), 413


@app.get("/health")
def health():
    return {"status": "ok", "gatekeeper": gatekeeper is not None}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
