"""
Shared image pre-processing and the ImageNet "gatekeeper", used by app.py and metrics.py,
so the web app and the evaluation see exactly the same inputs.
"""
import numpy as np
import torch
from PIL import Image, ImageOps

INPUT_SIZE = 128           # must match training: transforms.Resize((128, 128))
CLASSES = ("Cat", "Dog")   # ImageFolder sorts folders alphabetically: Cat=0, Dog=1

# ImageNet-1k class indices that count as "a dog or a cat"
DOG_CLASSES = list(range(151, 269))   # 118 dog breeds, Chihuahua ... Mexican hairless
CAT_CLASSES = list(range(281, 286))   # tabby, tiger cat, Persian, Siamese, Egyptian cat
PET_CLASSES = DOG_CLASSES + CAT_CLASSES

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_image(source) -> Image.Image:
    """Open a path or file-like object, apply the EXIF rotation, return RGB."""
    img = Image.open(source)
    img = ImageOps.exif_transpose(img)   # phone photos are often stored sideways
    return img.convert("RGB")            # handles PNG alpha, grayscale, CMYK


def cnn_tensor(img: Image.Image) -> torch.Tensor:
    """3 x 128 x 128 float tensor in [0, 1]: same as Resize((128,128)) + ToTensor()."""
    resized = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return torch.from_numpy(np.ascontiguousarray(arr))


def gate_tensor(img: Image.Image) -> torch.Tensor:
    """Standard ImageNet evaluation transform: shorter side to 256, centre crop 224, normalise."""
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((max(224, round(w * scale)), max(224, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(np.ascontiguousarray(arr.transpose(2, 0, 1)))


class Gatekeeper:
    """
    A small pretrained ImageNet model (MobileNetV3-Small) that decides whether a photo
    shows a dog or a cat at all, before the dog-vs-cat CNN is asked which one.

    pet probability = total softmax probability of the 118 dog + 5 cat ImageNet classes.
    """

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

    def check(self, img: Image.Image):
        """Return (is_pet, pet_probability, name of the most likely ImageNet class)."""
        pet, top = self.pet_probability(gate_tensor(img).unsqueeze(0))
        p = pet.item()
        return p >= self.threshold, p, self.categories[top.item()]
