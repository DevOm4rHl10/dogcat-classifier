"""
Check that the production path (ONNX Runtime + NumPy, serving.py) gives the same answers
as the original PyTorch model on real photos: same label, same confidence, same Grad-CAM.

Usage (from the project root):  python scripts/verify_onnx.py [--data data/test] [--n 200]
"""
import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from classifier import DogCatCNN  # noqa: E402
from inference import Gatekeeper as TorchGatekeeper  # noqa: E402
from serving import CLASSES, DogCatModel, Gatekeeper, cnn_array, gate_array, load_image  # noqa: E402

MODELS = ROOT / "saved_models"


def torch_gradcam(model, x):
    """The original PyTorch Grad-CAM (autograd), for comparison."""
    with torch.no_grad():
        feats = model.features(x)
    feats = feats.detach().requires_grad_(True)
    logit = model.classify(feats)
    p_dog = torch.sigmoid(logit).item()
    is_dog = p_dog > 0.5
    (logit if is_dog else -logit).sum().backward()
    weights = feats.grad[0].mean(dim=(1, 2))
    cam = F.relu((weights[:, None, None] * feats.detach()[0]).sum(0))
    cam = F.interpolate(cam[None, None], size=(128, 128), mode="bilinear", align_corners=False)[0, 0]
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return CLASSES[int(is_dog)], (p_dog if is_dog else 1 - p_dog), cam.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data" / "test"))
    parser.add_argument("--n", type=int, default=200, help="number of photos to compare")
    args = parser.parse_args()

    paths = sorted(p for p in Path(args.data).rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
    if not paths:
        raise SystemExit(f"No images under {args.data}")
    random.Random(0).shuffle(paths)
    paths = paths[:args.n]

    model = DogCatCNN()
    model.load_state_dict(torch.load(MODELS / "dogcat_cnn.pth", map_location="cpu", weights_only=True))
    model.eval()
    onnx_model = DogCatModel(str(MODELS / "cnn_features.onnx"), str(MODELS / "cnn_head.npz"))
    tgate = TorchGatekeeper(str(MODELS / "gatekeeper_mobilenet_v3_small.pth"), torch.device("cpu"))
    ogate = Gatekeeper(str(MODELS / "gatekeeper.onnx"), str(MODELS / "gatekeeper_classes.json"))

    same_label, conf_err, cam_err, cam_corr, gate_err = 0, [], [], [], []
    for p in paths:
        img = load_image(p)
        x = cnn_array(img)
        t_label, t_conf, t_cam = torch_gradcam(model, torch.from_numpy(x)[None])
        o_label, o_conf, o_cam = onnx_model.predict_with_gradcam(x)
        same_label += t_label == o_label
        conf_err.append(abs(t_conf - o_conf))
        cam_err.append(float(np.abs(t_cam - o_cam).mean()))
        if t_cam.std() > 0 and o_cam.std() > 0:
            cam_corr.append(float(np.corrcoef(t_cam.ravel(), o_cam.ravel())[0, 1]))
        t_pet = tgate.pet_probability(torch.from_numpy(gate_array(img))[None])[0].item()
        o_pet = ogate.check(img)[1]
        gate_err.append(abs(t_pet - o_pet))

    n = len(paths)
    print(f"Compared {n} photos from {args.data}")
    print(f"  same label:                {same_label}/{n}")
    print(f"  confidence, max difference: {max(conf_err):.4f}  (mean {np.mean(conf_err):.5f})")
    print(f"  Grad-CAM, mean abs diff:    {np.mean(cam_err):.4f}  (worst photo {max(cam_err):.4f})")
    print(f"  Grad-CAM, correlation:      mean {np.mean(cam_corr):.4f}, lowest {min(cam_corr):.4f}")
    print(f"  gatekeeper, max difference: {max(gate_err):.4f}")

    ok = same_label >= n - 1 and max(conf_err) < 0.01 and np.mean(cam_corr) > 0.98 and max(gate_err) < 0.01
    print("\nPASS: the web app matches PyTorch." if ok else
          "\nCHECK: some differences are larger than expected. Paste this output to investigate.")


if __name__ == "__main__":
    main()
