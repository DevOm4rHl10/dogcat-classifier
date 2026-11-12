"""
Export the models for production (the web app runs without PyTorch).

Reads   saved_models/dogcat_cnn.pth
        saved_models/gatekeeper_mobilenet_v3_small.pth   (scripts/download_gatekeeper.py)
Writes  saved_models/cnn_features.onnx       conv layers           (~0.4 MB)
        saved_models/cnn_head.npz            fc1 + fc2 weights      (~34 MB, float16)
        saved_models/gatekeeper.onnx         MobileNetV3-Small      (~10 MB)
        saved_models/gatekeeper_classes.json ImageNet class names

Usage (from the project root, with requirements-train.txt installed):
    python scripts/export_onnx.py
Then check the exported models against PyTorch on real photos:
    python scripts/verify_onnx.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from classifier import DogCatCNN  # noqa: E402

MODELS = ROOT / "saved_models"


class Features(torch.nn.Module):
    """Just the convolutional part of the CNN, up to the activations Grad-CAM uses."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model.features(x)


def export(module, example, path: Path, input_name: str, output_name: str):
    kwargs = dict(input_names=[input_name], output_names=[output_name])
    try:
        torch.onnx.export(module, (example,), str(path), dynamo=True, **kwargs)
    except TypeError:   # older PyTorch without the `dynamo` argument
        torch.onnx.export(module, (example,), str(path), opset_version=17, **kwargs)

    # Newer PyTorch writes the weights to a separate "<name>.data" file. Put them back
    # inside the .onnx file, so a single file is all the server needs.
    import onnx

    m = onnx.load(str(path))                    # also reads any external .data file
    for leftover in path.parent.glob(path.name + "*.data"):
        leftover.unlink()
    for leftover in path.parent.glob(path.stem + "*.data"):
        leftover.unlink()
    onnx.save_model(m, str(path), save_as_external_data=False)


def check(path: Path, module, example):
    """Compare ONNX Runtime with PyTorch on the example input."""
    import onnxruntime as ort

    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, {sess.get_inputs()[0].name: example.numpy()})[0]
    with torch.no_grad():
        want = module(example).numpy()
    err = float(np.abs(got - want).max())
    print(f"  {path.name}: {path.stat().st_size / 1e6:.1f} MB, max difference vs PyTorch {err:.1e}")
    if err > 1e-3:
        raise SystemExit(f"{path.name} doesn't match PyTorch closely enough.")


def main():
    torch.manual_seed(0)

    # The dog-vs-cat CNN: conv layers to ONNX, head weights to NumPy
    model = DogCatCNN()
    model.load_state_dict(torch.load(MODELS / "dogcat_cnn.pth", map_location="cpu", weights_only=True))
    model.eval()
    features = Features(model).eval()
    x = torch.rand(1, 3, 128, 128)
    out = MODELS / "cnn_features.onnx"
    export(features, x, out, "image", "features")
    check(out, features, x)

    head = MODELS / "cnn_head.npz"
    np.savez(head,
             fc1_weight=model.fc1.weight.detach().numpy().astype(np.float16),
             fc1_bias=model.fc1.bias.detach().numpy().astype(np.float32),
             fc2_weight=model.fc2.weight.detach().numpy().astype(np.float32),
             fc2_bias=model.fc2.bias.detach().numpy().astype(np.float32))
    print(f"  {head.name}: {head.stat().st_size / 1e6:.1f} MB (fc1 stored as float16)")

    # The gatekeeper
    from torchvision.models import mobilenet_v3_small

    ckpt = torch.load(MODELS / "gatekeeper_mobilenet_v3_small.pth", map_location="cpu", weights_only=True)
    gate = mobilenet_v3_small()
    gate.load_state_dict(ckpt["state_dict"])
    gate.eval()
    g = torch.rand(1, 3, 224, 224)
    out = MODELS / "gatekeeper.onnx"
    export(gate, g, out, "image", "logits")
    check(out, gate, g)
    (MODELS / "gatekeeper_classes.json").write_text(json.dumps(list(ckpt["categories"])))
    print("  gatekeeper_classes.json written")

    print("\nDone. Next: python scripts/verify_onnx.py")


if __name__ == "__main__":
    main()
