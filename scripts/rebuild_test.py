"""
Rebuild data/test from the raw Kaggle images when train/ and val/ still exist.

The test set is everything in the raw dataset that is NOT in train or val. Matching is done
by file content (MD5), so a rebuilt test set is leak-free by construction: no photo whose
exact bytes appear in train or val can end up in it, whatever the old split script did.
Unreadable images are skipped, as the training data loader would have done.

Usage (from the project root):
  python scripts/rebuild_test.py --raw "/path/to/PetImages"     # folder with Cat/ and Dog/
  python scripts/check_leakage.py                               # optional: also checks near-duplicates
  python metrics.py
"""
import argparse
import hashlib
import shutil
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
CLASSES = ("Cat", "Dog")


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def readable(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def images(folder: Path):
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw", required=True, help="raw dataset folder containing Cat/ and Dog/")
    parser.add_argument("--data", default=str(ROOT / "data"))
    args = parser.parse_args()
    raw, data = Path(args.raw), Path(args.data)

    for cls in CLASSES:
        if not (raw / cls).is_dir():
            raise SystemExit(f"Expected {raw / cls}. Point --raw at the folder that contains Cat/ and Dog/.")
        if not (data / "train" / cls).is_dir():
            raise SystemExit(f"Expected {data / 'train' / cls}: the test set can only be rebuilt if train/ still exists.")

    print("Hashing train and val ...")
    used = set()
    for split in ("train", "val"):
        for cls in CLASSES:
            folder = data / split / cls
            if folder.is_dir():
                used.update(md5(p) for p in images(folder))
    print(f"  {len(used):,} distinct images in train + val")

    test = data / "test"
    if test.exists():
        raise SystemExit(f"{test} already exists; move or delete it first so nothing is mixed.")

    for cls in CLASSES:
        dest = test / cls
        dest.mkdir(parents=True)
        kept = in_train_val = broken = dupes = 0
        seen = set()
        for p in images(raw / cls):
            h = md5(p)
            if h in used:
                in_train_val += 1
                continue
            if h in seen:                 # same photo twice in the raw data: keep one copy
                dupes += 1
                continue
            if not readable(p):
                broken += 1
                continue
            seen.add(h)
            shutil.copy2(p, dest / p.name)
            kept += 1
        print(f"{cls}: {kept:,} test images  (skipped {in_train_val:,} in train/val, "
              f"{dupes:,} repeated, {broken:,} unreadable)")

    print(f"\nRebuilt {test}. Expect about 2,500 per class if train/val came from one clean 60/20/20 split.")


if __name__ == "__main__":
    main()
