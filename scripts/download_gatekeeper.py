"""
Download the pretrained MobileNetV3-Small ImageNet weights once and store them in the repo,
so the deployed app never downloads anything at start-up.

Usage (from the project root):  python scripts/download_gatekeeper.py
Output: saved_models/gatekeeper_mobilenet_v3_small.pth  (~10 MB)
"""
from pathlib import Path

import torch
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

OUT = Path(__file__).resolve().parent.parent / "saved_models" / "gatekeeper_mobilenet_v3_small.pth"

weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
model = mobilenet_v3_small(weights=weights)

OUT.parent.mkdir(exist_ok=True)
torch.save({"state_dict": model.state_dict(), "categories": list(weights.meta["categories"])}, OUT)
print(f"Saved {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")
