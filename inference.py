"""
PyTorch-side helpers for evaluation (metrics.py) and export (scripts/export_onnx.py).

Pre-processing is imported from serving.py, so the PyTorch evaluation and the
ONNX-based web app always see exactly the same inputs.
"""
import torch

from serving import CLASSES, INPUT_SIZE, PET_CLASSES, cnn_array, gate_array, load_image  # noqa: F401


def cnn_tensor(img) -> torch.Tensor:
    """3 x 128 x 128 float tensor in [0, 1]: same as Resize((128,128)) + ToTensor()."""
    return torch.from_numpy(cnn_array(img))


def gate_tensor(img) -> torch.Tensor:
    """3 x 224 x 224 ImageNet-normalised tensor for the gatekeeper."""
    return torch.from_numpy(gate_array(img))


class Gatekeeper:
    """PyTorch version of the MobileNetV3-Small gatekeeper (see serving.Gatekeeper)."""

    def __init__(self, path: str, device: torch.device, threshold: float = 0.2):
        from torchvision.models import mobilenet_v3_small

        ckpt = torch.load(path, map_location=device, weights_only=True)
        self.model = mobilenet_v3_small()          # architecture only, no download
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.to(device).eval()
        self.categories = ckpt["categories"]
        self.device = device
        self.threshold = threshold
        self.pet_idx = torch.tensor(PET_CLASSES)

    @torch.no_grad()
    def pet_probability(self, batch: torch.Tensor):
        """batch: N x 3 x 224 x 224 -> (pet probability [N], top ImageNet class index [N])."""
        probs = torch.softmax(self.model(batch.to(self.device)), dim=1).cpu()
        return probs[:, self.pet_idx].sum(dim=1), probs.argmax(dim=1)
