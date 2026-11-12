"""
Production inference: ONNX Runtime + NumPy, no PyTorch.

PyTorch alone needs ~300 MB of memory, which doesn't fit a 512 MB free server, so the
web app serves exported models instead (scripts/export_onnx.py):

  cnn_features.onnx   the CNN's convolutional layers (image -> last conv activations)
  cnn_head.npz        the fully connected head's weights, run in NumPy
  gatekeeper.onnx     MobileNetV3-Small (ImageNet), the "neither" check

The head (max-pool -> fc1 -> ReLU -> fc2) is small enough to run, and differentiate, in
NumPy. That gives Grad-CAM its gradients without an autograd library.

Pre-processing lives here too, and the PyTorch code (inference.py) imports it, so
training-side evaluation and the web app always see identical inputs.
"""
import json

import numpy as np
from PIL import Image, ImageOps

INPUT_SIZE = 128           # must match training: transforms.Resize((128, 128))
CLASSES = ("Cat", "Dog")   # ImageFolder sorts folders alphabetically: Cat=0, Dog=1

# ImageNet-1k class indices that count as "a dog or a cat"
DOG_CLASSES = list(range(151, 269))   # 118 dog breeds, Chihuahua ... Mexican hairless
CAT_CLASSES = list(range(281, 286))   # tabby, tiger cat, Persian, Siamese, Egyptian cat
PET_CLASSES = DOG_CLASSES + CAT_CLASSES

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Pre-processing (shared with the PyTorch code)
# ---------------------------------------------------------------------------
def load_image(source) -> Image.Image:
    """Open a path or file-like object, apply the EXIF rotation, return RGB."""
    img = Image.open(source)
    img = ImageOps.exif_transpose(img)   # phone photos are often stored sideways
    return img.convert("RGB")            # handles PNG alpha, grayscale, CMYK


def cnn_array(img: Image.Image) -> np.ndarray:
    """3 x 128 x 128 float32 in [0, 1]: same as Resize((128,128)) + ToTensor()."""
    resized = img.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    arr = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
    return np.ascontiguousarray(arr)


def gate_array(img: Image.Image) -> np.ndarray:
    """Standard ImageNet evaluation transform: shorter side to 256, centre crop 224, normalise."""
    w, h = img.size
    scale = 256 / min(w, h)
    img = img.resize((max(224, round(w * scale)), max(224, round(h * scale))), Image.BILINEAR)
    w, h = img.size
    left, top = (w - 224) // 2, (h - 224) // 2
    img = img.crop((left, top, left + 224, top + 224))
    arr = (np.asarray(img, dtype=np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(arr.transpose(2, 0, 1))


# ---------------------------------------------------------------------------
# ONNX Runtime sessions
# ---------------------------------------------------------------------------
def _session(path: str, threads: int):
    import onnxruntime as ort   # imported here so the training code doesn't need it

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])


class DogCatModel:
    """The dog-vs-cat CNN: conv layers in ONNX Runtime, head and Grad-CAM in NumPy."""

    def __init__(self, features_path: str, head_path: str, threads: int = 1):
        self.session = _session(features_path, threads)
        self.input_name = self.session.get_inputs()[0].name
        head = np.load(head_path)
        self.w1 = head["fc1_weight"].astype(np.float32)            # 512 x 32768
        self.b1 = head["fc1_bias"].astype(np.float32)              # 512
        self.w2 = head["fc2_weight"].astype(np.float32).reshape(-1)  # 512
        self.b2 = float(head["fc2_bias"].reshape(-1)[0])

    def features(self, x: np.ndarray) -> np.ndarray:
        """3 x 128 x 128 -> C x 32 x 32 activations of the last conv block (after ReLU)."""
        return self.session.run(None, {self.input_name: x[None]})[0][0]

    def predict_with_gradcam(self, x: np.ndarray):
        """Return (label, confidence, cam): cam is INPUT_SIZE x INPUT_SIZE in [0, 1]."""
        acts = self.features(x)
        label, confidence, grads = self.head_and_gradients(acts)
        return label, confidence, gradcam(acts, grads)

    def head_and_gradients(self, acts: np.ndarray):
        """
        Forward through the head, then back-propagate by hand to the conv activations.

        Forward:  pooled = maxpool2x2(acts); z1 = W1 . flatten(pooled) + b1
                  logit = w2 . relu(z1) + b2;   P(dog) = sigmoid(logit)
        Backward: Grad-CAM explains the predicted class, so the gradient is taken of
                  +logit for a dog and -logit for a cat (the logit, not the sigmoid,
                  avoids vanishing gradients on confident images).
        """
        C, H, W = acts.shape
        # 2x2 windows as a last axis of 4, in row-major order (PyTorch's tie-breaking order)
        win = acts.reshape(C, H // 2, 2, W // 2, 2).transpose(0, 1, 3, 2, 4).reshape(C, H // 2, W // 2, 4)
        idx = win.argmax(axis=-1)
        pooled = np.take_along_axis(win, idx[..., None], axis=-1)[..., 0]

        z1 = self.w1 @ pooled.reshape(-1) + self.b1
        logit = float(self.w2 @ np.maximum(z1, 0) + self.b2)
        p_dog = 1.0 / (1.0 + np.exp(-logit))
        is_dog = p_dog > 0.5

        sign = 1.0 if is_dog else -1.0
        g_z1 = sign * self.w2 * (z1 > 0)                         # through fc2 and ReLU
        g_pooled = (self.w1.T @ g_z1).reshape(C, H // 2, W // 2)  # through fc1 and flatten
        g_win = np.zeros_like(win)
        np.put_along_axis(g_win, idx[..., None], g_pooled[..., None], axis=-1)  # max-pool routes to the max
        grads = g_win.reshape(C, H // 2, W // 2, 2, 2).transpose(0, 1, 3, 2, 4).reshape(C, H, W)

        label = CLASSES[1] if is_dog else CLASSES[0]
        confidence = p_dog if is_dog else 1.0 - p_dog
        return label, float(confidence), grads


def gradcam(acts: np.ndarray, grads: np.ndarray) -> np.ndarray:
    """Weight each feature map by its average gradient, keep the positive part, upsample, normalise."""
    weights = grads.mean(axis=(1, 2))
    cam = np.maximum((weights[:, None, None] * acts).sum(axis=0), 0).astype(np.float32)
    cam = np.asarray(Image.fromarray(cam, mode="F").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR))
    cam = cam - cam.min()
    return cam / (cam.max() + 1e-8)


class Gatekeeper:
    """
    MobileNetV3-Small (ImageNet) decides whether a photo shows a dog or a cat at all.
    pet probability = total softmax probability of the 118 dog + 5 cat ImageNet classes.
    """

    def __init__(self, onnx_path: str, classes_path: str, threshold: float = 0.2, threads: int = 1):
        self.session = _session(onnx_path, threads)
        self.input_name = self.session.get_inputs()[0].name
        with open(classes_path) as f:
            self.categories = json.load(f)
        self.threshold = threshold

    def check(self, img: Image.Image):
        """Return (is_pet, pet_probability, name of the most likely ImageNet class)."""
        logits = self.session.run(None, {self.input_name: gate_array(img)[None]})[0][0]
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        p = float(probs[PET_CLASSES].sum())
        return p >= self.threshold, p, self.categories[int(probs.argmax())]
