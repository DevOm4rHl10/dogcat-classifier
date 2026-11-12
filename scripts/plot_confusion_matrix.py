"""
Draw the confusion matrix from static/metrics/metrics.json as a PNG for the README.
Uses only Pillow (no matplotlib). metrics.py runs this automatically after each evaluation.

Usage (from the project root):  python scripts/plot_confusion_matrix.py
Output: docs/confusion_matrix.png
"""
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
METRICS = ROOT / "static" / "metrics" / "metrics.json"
OUT = ROOT / "docs" / "confusion_matrix.png"

# Same palette as the web app
SURFACE = (255, 255, 255)
INK = (23, 33, 58)
MUTED = (91, 100, 120)
COLD = (43, 63, 191)
S = 2   # render at 2x so it stays sharp on high-density screens


def font(size, bold=False):
    names = (["DejaVuSans-Bold.ttf", "LiberationSans-Bold.ttf", "Arial Bold.ttf"] if bold
             else ["DejaVuSans.ttf", "LiberationSans-Regular.ttf", "Arial.ttf"])
    for name in names:
        try:
            return ImageFont.truetype(name, size * S)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size * S)   # Pillow >= 10.1
    except TypeError:
        return ImageFont.load_default()


def mix(share):
    """Tint between white and the brand blue; share in [0, 1]."""
    t = 0.06 + 0.74 * share
    return tuple(round(SURFACE[i] + (COLD[i] - SURFACE[i]) * t) for i in range(3)), t > 0.45


def text(draw, xy, s, f, fill, anchor="mm"):
    draw.text((xy[0] * S, xy[1] * S), s, font=f, fill=fill, anchor=anchor)


def render(report, out=OUT):
    cm = report["confusion_matrix"]["rows_true_cols_pred"]
    labels = report["confusion_matrix"]["labels"]
    n, acc = report["n"], report["accuracy"]
    peak = max(max(row) for row in cm)

    W, H = 600, 490
    cell_w, cell_h, gap = 200, 150, 4
    left, top = 150, 130
    img = Image.new("RGB", (W * S, H * S), SURFACE)
    d = ImageDraw.Draw(img)

    text(d, (32, 34), "Confusion matrix", font(26, True), INK, "lm")
    text(d, (32, 66), f"{n:,} held-out test photos, accuracy {acc * 100:.1f}%", font(15), MUTED, "lm")

    for j, lab in enumerate(labels):
        text(d, (left + j * (cell_w + gap) + cell_w / 2, top - 20), f"Said {lab.lower()}", font(15, True), MUTED)

    for i, row in enumerate(cm):
        y = top + i * (cell_h + gap)
        text(d, (left - 16, y + cell_h / 2), f"Real {labels[i].lower()}", font(15, True), MUTED, "rm")
        total = sum(row)
        for j, v in enumerate(row):
            x = left + j * (cell_w + gap)
            colour, dark = mix(v / peak)
            d.rounded_rectangle([x * S, y * S, (x + cell_w) * S, (y + cell_h) * S], radius=6 * S, fill=colour)
            fg = SURFACE if dark else INK
            text(d, (x + cell_w / 2, y + cell_h / 2 - 12), f"{v:,}", font(34, True), fg)
            text(d, (x + cell_w / 2, y + cell_h / 2 + 24), f"{100 * v / total:.1f}% of {labels[i].lower()}s", font(14), fg)

    text(d, (32, H - 26), "Rows: true class. Columns: model's prediction (threshold 0.5).", font(13), MUTED, "lm")

    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out, optimize=True)
    return out


if __name__ == "__main__":
    if not METRICS.exists():
        raise SystemExit(f"{METRICS} not found: run python metrics.py first.")
    print(f"Saved {render(json.loads(METRICS.read_text()))}")
