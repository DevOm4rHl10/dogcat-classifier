"""
Check that no test (or validation) photo also appears in the training set.

Three checks, from strictest to loosest:
  1. same file name in two splits   -> the split script ran more than once with a different shuffle
  2. byte-identical files           -> exact copies, including duplicates inside the Kaggle dataset
  3. near-duplicates (perceptual hash) -> the same photo resized, re-saved or slightly edited

Usage (from the project root):
  python scripts/check_leakage.py                       # report only
  python scripts/check_leakage.py --write-clean          # also build data/test_clean without leaked photos
  python metrics.py --data data/test_clean               # then re-measure on the clean set
"""
import argparse
import hashlib
import os
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLITS = ("train", "val", "test")
POPCOUNT = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def list_split(data: Path, split: str):
    """[(path, class_name)] for every image in data/<split>/<class>/"""
    items = []
    base = data / split
    if not base.is_dir():
        return items
    for cls_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        items += [(p, cls_dir.name) for p in sorted(cls_dir.iterdir()) if p.suffix.lower() in IMAGE_EXTS]
    return items


def md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def dhash(path: Path):
    """64-bit difference hash: robust to resizing, re-compression and small colour changes."""
    try:
        with Image.open(path) as im:
            g = np.asarray(im.convert("L").resize((9, 8), Image.BILINEAR), dtype=np.int16)
    except Exception:
        return None
    return np.packbits((g[:, 1:] > g[:, :-1]).flatten())   # 8 bytes


def near_duplicates(query, reference, max_distance, chunk=256):
    """For each query hash, the closest reference hash within max_distance bits: {q_index: (r_index, distance)}"""
    found = {}
    if len(query) == 0 or len(reference) == 0:
        return found
    for start in range(0, len(query), chunk):
        q = query[start:start + chunk]
        dist = POPCOUNT[q[:, None, :] ^ reference[None, :, :]].sum(axis=2, dtype=np.int32)
        best = dist.argmin(axis=1)
        best_d = dist[np.arange(len(q)), best]
        for i in np.where(best_d <= max_distance)[0]:
            found[start + i] = (int(best[i]), int(best_d[i]))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default=str(ROOT / "data"))
    parser.add_argument("--max-distance", type=int, default=4,
                        help="bits out of 64; 0-4 is almost always the same photo (default 4)")
    parser.add_argument("--write-clean", action="store_true",
                        help="create <data>/test_clean with leaked test photos removed (hard links, no extra space)")
    args = parser.parse_args()
    data = Path(args.data)

    splits = {s: list_split(data, s) for s in SPLITS}
    for s, items in splits.items():
        per_class = defaultdict(int)
        for _, c in items:
            per_class[c] += 1
        print(f"{s:>5}: {len(items):6,} images  " + "  ".join(f"{c}={n:,}" for c, n in sorted(per_class.items())))
    if not splits["train"] or not splits["test"]:
        raise SystemExit(f"Need {data}/train and {data}/test.")

    leaked = defaultdict(set)   # test path -> reasons

    # 1. Same (class, file name) in two splits
    print("\n1. Same file name in two splits")
    names = {s: {(c, p.name): p for p, c in items} for s, items in splits.items()}
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        common = names[a].keys() & names[b].keys()
        print(f"   {a} & {b}: {len(common):,}")
        if b == "test":
            for key in common:
                leaked[names["test"][key]].add(f"same name as {a}")

    # 2. Byte-identical files
    print("\n2. Byte-identical files (hashing every image, this takes a moment)")
    hashes = {s: {} for s in SPLITS}
    for s, items in splits.items():
        for p, c in items:
            hashes[s].setdefault(md5(p), []).append(p)
    for a, b in (("train", "test"), ("train", "val"), ("val", "test")):
        common = hashes[a].keys() & hashes[b].keys()
        n = sum(len(hashes[b][h]) for h in common)
        print(f"   {b} files identical to a {a} file: {n:,}")
        if b == "test":
            for h in common:
                for p in hashes["test"][h]:
                    leaked[p].add(f"identical to {a}")
    within = sum(len(v) - 1 for v in hashes["test"].values() if len(v) > 1)
    print(f"   duplicates inside test itself: {within:,} (harmless for leakage, just double-counted)")

    # 3. Near-duplicates against train + val
    print(f"\n3. Near-duplicates (perceptual hash, <= {args.max_distance} of 64 bits different)")
    ref_items = splits["train"] + splits["val"]
    ref_hash, ref_paths = [], []
    for p, c in ref_items:
        h = dhash(p)
        if h is not None:
            ref_hash.append(h); ref_paths.append(p)
    test_hash, test_paths = [], []
    for p, c in splits["test"]:
        h = dhash(p)
        if h is not None:
            test_hash.append(h); test_paths.append(p)
    found = near_duplicates(np.array(test_hash), np.array(ref_hash), args.max_distance)
    print(f"   test photos with a near-twin in train/val: {len(found):,}")
    for qi, (ri, d) in found.items():
        leaked[test_paths[qi]].add(f"near-duplicate of {ref_paths[ri].relative_to(data)} (distance {d})")

    # Report
    total_test = len(splits["test"])
    print(f"\nRESULT: {len(leaked):,} of {total_test:,} test photos ({100 * len(leaked) / total_test:.2f}%) "
          f"also appear in train/val in some form.")
    report = data / "leakage_report.txt"
    with open(report, "w") as f:
        for p in sorted(leaked):
            f.write(f"{p.relative_to(data)}\t{'; '.join(sorted(leaked[p]))}\n")
    print(f"Details: {report}")
    if leaked:
        print("Spot-check a few pairs from the report by eye: at distance 3-4 a few may be different photos "
              "that just look alike (e.g. plain backgrounds).")

    if args.write_clean:
        clean = data / "test_clean"
        shutil.rmtree(clean, ignore_errors=True)
        kept = 0
        for p, c in splits["test"]:
            if p in leaked:
                continue
            dest = clean / c / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(p, dest)          # hard link: no extra disk space
            except OSError:
                shutil.copy2(p, dest)
            kept += 1
        print(f"\nWrote {clean} with {kept:,} photos. Re-measure with:  python metrics.py --data {clean}")


if __name__ == "__main__":
    main()
