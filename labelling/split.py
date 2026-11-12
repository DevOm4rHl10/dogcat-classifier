"""
Split the raw Kaggle images into train / val / test folders.

Fixes over the first version:
  * file order is sorted before shuffling, so the split is identical on every machine
  * the output folder is rebuilt from scratch, so re-running never leaks files across splits
  * corrupt files and non-images (e.g. Thumbs.db) are skipped instead of copied
  * exact duplicate images are dropped, so the same picture can't land in train and test
  * paths are resolved from this file, not from wherever the script is launched

Usage:  python labelling/split.py
"""
import hashlib
import random
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "training-data"   # training-data/<Class>/*.jpg
OUTPUT_DIR = ROOT / "data"            # data/{train,val,test}/<Class>/

RATIOS = {"train": 0.6, "val": 0.2, "test": 0.2}
SEED = 42
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def is_valid_image(path: Path) -> bool:
    if path.suffix.lower() not in IMAGE_EXTS:
        return False
    try:
        with Image.open(path) as im:
            im.verify()  # catches truncated / corrupt files
        return True
    except Exception:
        return False


def file_hash(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def main() -> None:
    assert abs(sum(RATIOS.values()) - 1.0) < 1e-9, "ratios must sum to 1"
    classes = sorted(p.name for p in SOURCE_DIR.iterdir() if p.is_dir())

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    rng = random.Random(SEED)
    seen_hashes: set[str] = set()

    for cls in classes:
        files = sorted(p for p in (SOURCE_DIR / cls).iterdir() if p.is_file())
        valid, skipped, dupes = [], 0, 0
        for f in files:
            if not is_valid_image(f):
                skipped += 1
                continue
            h = file_hash(f)
            if h in seen_hashes:   # also catches the same image filed under both classes
                dupes += 1
                continue
            seen_hashes.add(h)
            valid.append(f)

        rng.shuffle(valid)
        n = len(valid)
        n_train = int(RATIOS["train"] * n)
        n_val = int(RATIOS["val"] * n)
        splits = {
            "train": valid[:n_train],
            "val": valid[n_train:n_train + n_val],
            "test": valid[n_train + n_val:],
        }

        for split, split_files in splits.items():
            dest = OUTPUT_DIR / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, dest / f.name)

        counts = ", ".join(f"{k}={len(v)}" for k, v in splits.items())
        print(f"{cls}: {counts}  (skipped {skipped} unreadable, {dupes} duplicates)")

    print(f"Done. Output in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
