"""
Evaluate the trained CNN (and the gatekeeper, if present) on the held-out test set.

Writes static/metrics/metrics.json and thumbnails of the hardest mistakes;
the web app shows them at /metrics.

Usage (from the project root):  python metrics.py
                                python metrics.py --data data/test --threshold 0.5
"""
import argparse
import datetime
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "static" / "metrics"
MODEL_PATH = ROOT / "saved_models" / "dogcat_cnn.pth"
GATE_PATH = ROOT / "saved_models" / "gatekeeper_mobilenet_v3_small.pth"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
GATE_THRESHOLDS = [0.05, 0.1, 0.2, 0.3, 0.5]


# ---------------------------------------------------------------------------
# Metric computations (pure NumPy)
# ---------------------------------------------------------------------------
def safe_div(a, b):
    return float(a) / float(b) if b else 0.0


def roc_curve(scores, labels, max_points=200):
    """ROC for 'dog' as the positive class. Returns (points [[fpr, tpr, threshold]...], auc)."""
    order = np.argsort(-scores, kind="mergesort")
    s, y = scores[order], labels[order]
    distinct = np.where(np.diff(s))[0]                 # last index of each tie group
    idx = np.r_[distinct, len(s) - 1]
    tps = np.cumsum(y)[idx]
    fps = (idx + 1) - tps
    tpr = np.r_[0.0, tps / max(y.sum(), 1)]
    fpr = np.r_[0.0, fps / max((1 - y).sum(), 1)]
    thr = np.r_[1.0, s[idx]]
    trapezoid = getattr(np, "trapezoid", None) or np.trapz   # NumPy 2 renamed trapz
    auc = float(trapezoid(tpr, fpr))
    keep = np.unique(np.linspace(0, len(fpr) - 1, min(max_points, len(fpr))).round().astype(int))
    points = [[round(float(fpr[i]), 4), round(float(tpr[i]), 4), round(float(thr[i]), 4)] for i in keep]
    return points, auc


def reliability(p_dog, labels, threshold, edges=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0)):
    """Calibration: for predictions made with confidence in each band, how often were they right?"""
    preds = (p_dog > threshold).astype(int)
    conf = np.where(preds == 1, p_dog, 1 - p_dog)
    correct = preds == labels
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf >= lo) & ((conf < hi) if hi < 1.0 else (conf <= hi))
        n = int(mask.sum())
        bins.append({
            "lo": lo, "hi": hi, "count": n,
            "mean_confidence": round(float(conf[mask].mean()), 4) if n else None,
            "accuracy": round(float(correct[mask].mean()), 4) if n else None,
        })
    return bins


def compute_report(p_dog, labels, threshold=0.5):
    p_dog = np.asarray(p_dog, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    pred = (p_dog > threshold).astype(np.int64)

    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    tp = int(((y == 1) & (pred == 1)).sum())

    def scores(prec, rec, support):
        f1 = safe_div(2 * prec * rec, prec + rec)
        return {"precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "support": support}

    per_class = {
        "Cat": scores(safe_div(tn, tn + fn), safe_div(tn, tn + fp), tn + fp),
        "Dog": scores(safe_div(tp, tp + fp), safe_div(tp, tp + fn), fn + tp),
    }
    macro = {k: round((per_class["Cat"][k] + per_class["Dog"][k]) / 2, 4) for k in ("precision", "recall", "f1")}

    roc_points, auc = roc_curve(p_dog, y)
    eps = 1e-7
    clipped = np.clip(p_dog, eps, 1 - eps)
    log_loss = float(-np.mean(y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
    conf = np.where(pred == 1, p_dog, 1 - p_dog)
    wrong = pred != y

    return {
        "threshold": threshold,
        "n": int(len(y)),
        "accuracy": round(safe_div(tp + tn, len(y)), 4),
        "confusion_matrix": {"labels": ["Cat", "Dog"], "rows_true_cols_pred": [[tn, fp], [fn, tp]]},
        "per_class": per_class,
        "macro": macro,
        "roc": {"auc": round(auc, 4), "points": roc_points},
        "log_loss": round(log_loss, 4),
        "brier": round(float(np.mean((p_dog - y) ** 2)), 4),
        "mean_confidence_correct": round(float(conf[~wrong].mean()), 4) if (~wrong).any() else None,
        "mean_confidence_wrong": round(float(conf[wrong].mean()), 4) if wrong.any() else None,
        "reliability": reliability(p_dog, y, threshold),
    }


def gate_report(pet_prob, labels, threshold):
    """How often the gatekeeper wrongly turns away a real test cat or dog."""
    pet_prob, y = np.asarray(pet_prob), np.asarray(labels)
    rows = []
    for t in sorted(set(GATE_THRESHOLDS + [threshold])):
        rejected = pet_prob < t
        rows.append({
            "threshold": t,
            "rejected_all": round(float(rejected.mean()), 4),
            "rejected_cat": round(float(rejected[y == 0].mean()), 4) if (y == 0).any() else 0.0,
            "rejected_dog": round(float(rejected[y == 1].mean()), 4) if (y == 1).any() else 0.0,
        })
    return {"threshold": threshold, "rows": rows}


# ---------------------------------------------------------------------------
# Data loading (used with torch's DataLoader; kept free of torch imports at module level)
# ---------------------------------------------------------------------------
class TestImages:
    def __init__(self, root: Path, with_gate: bool):
        from inference import CLASSES
        self.samples = []
        for label, cls in enumerate(CLASSES):
            folder = root / cls
            if not folder.is_dir():
                raise SystemExit(f"Expected a folder {folder} (classes must be {CLASSES}).")
            self.samples += [(p, label) for p in sorted(folder.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
        self.with_gate = with_gate

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        from inference import cnn_tensor, gate_tensor, load_image
        path, label = self.samples[i]
        try:
            img = load_image(path)
        except Exception:
            return None                                # unreadable file: skipped and counted
        gate = gate_tensor(img) if self.with_gate else None
        return cnn_tensor(img), gate, label, i


def collate(batch):
    import torch
    batch = [b for b in batch if b is not None]
    if not batch:
        return None
    cnn, gate, labels, idx = zip(*batch)
    return (torch.stack(cnn), torch.stack(gate) if gate[0] is not None else None,
            torch.tensor(labels), torch.tensor(idx))


def save_mistakes(samples, p_dog, labels, threshold, k=8):
    """Thumbnails of the k most confident wrong predictions."""
    from PIL import Image
    from inference import CLASSES, load_image

    out = OUT_DIR / "mistakes"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)

    pred = (p_dog > threshold).astype(int)
    conf = np.where(pred == 1, p_dog, 1 - p_dog)
    wrong = np.where(pred != labels)[0]
    worst = wrong[np.argsort(-conf[wrong])][:k]

    items = []
    for rank, i in enumerate(worst):
        img = load_image(samples[i][0])
        img.thumbnail((240, 240), Image.BILINEAR)
        name = f"{rank}.jpg"
        img.save(out / name, "JPEG", quality=80)
        items.append({"src": f"metrics/mistakes/{name}", "true": CLASSES[labels[i]],
                      "pred": CLASSES[pred[i]], "confidence": round(float(conf[i]), 4),
                      "file": samples[i][0].name})
    return items


# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "data" / "test"))
    parser.add_argument("--threshold", type=float, default=0.5, help="P(dog) above which we predict dog")
    parser.add_argument("--gate-threshold", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    import torch
    from PIL import ImageFile
    from torch.utils.data import DataLoader

    from classifier import DogCatCNN
    from inference import Gatekeeper

    ImageFile.LOAD_TRUNCATED_IMAGES = True   # same setting the model was trained with

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DogCatCNN().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
    model.eval()

    gate = Gatekeeper(str(GATE_PATH), device, args.gate_threshold) if GATE_PATH.exists() else None
    if gate is None:
        print("Gatekeeper weights not found; skipping its evaluation (run scripts/download_gatekeeper.py).")

    dataset = TestImages(Path(args.data), with_gate=gate is not None)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers, collate_fn=collate)
    print(f"Evaluating {len(dataset)} images from {args.data} on {device} ...")

    p_dog = np.full(len(dataset), np.nan)
    pet_prob = np.full(len(dataset), np.nan)
    labels = np.array([lbl for _, lbl in dataset.samples])

    with torch.no_grad():
        for b, batch in enumerate(loader):
            if batch is None:
                continue
            cnn, gate_in, _, idx = batch
            p_dog[idx.numpy()] = torch.sigmoid(model(cnn.to(device))).squeeze(1).cpu().numpy()
            if gate is not None:
                pet_prob[idx.numpy()] = gate.pet_probability(gate_in)[0].numpy()
            if b % 10 == 0:
                print(f"  batch {b + 1}/{len(loader)}", flush=True)

    ok = ~np.isnan(p_dog)
    skipped = int((~ok).sum())
    samples = [s for s, keep in zip(dataset.samples, ok) if keep]

    report = compute_report(p_dog[ok], labels[ok], args.threshold)
    report["skipped_unreadable"] = skipped
    report["gatekeeper"] = gate_report(pet_prob[ok], labels[ok], args.gate_threshold) if gate else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    report["mistakes"] = save_mistakes(samples, p_dog[ok], labels[ok], args.threshold)
    report["model_sha256"] = hashlib.sha256(MODEL_PATH.read_bytes()).hexdigest()[:12]
    report["generated"] = datetime.date.today().isoformat()

    (OUT_DIR / "metrics.json").write_text(json.dumps(report, indent=1))

    import sys
    sys.path.insert(0, str(ROOT / "scripts"))
    from plot_confusion_matrix import render
    print(f"Saved {render(report)}")

    cm = report["confusion_matrix"]["rows_true_cols_pred"]
    print(f"\nAccuracy {report['accuracy']:.4f} | ROC AUC {report['roc']['auc']:.4f} | skipped {skipped}")
    print(f"Confusion matrix (rows = true Cat/Dog, cols = predicted Cat/Dog): {cm}")
    for cls, s in report["per_class"].items():
        print(f"  {cls}: precision {s['precision']:.4f}  recall {s['recall']:.4f}  F1 {s['f1']:.4f}  (n={s['support']})")
    print(f"\nWrote {OUT_DIR / 'metrics.json'}. Start the app and open /metrics.")


if __name__ == "__main__":
    main()
