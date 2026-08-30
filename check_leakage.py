"""
check_leakage.py -- measure near-duplicate leakage between dataset subsets, for:

    "Leakage-Free Evaluation of Mask R-CNN for Fire Region Segmentation in
     Still Images: Accuracy and Failure Modes on Fire-Illuminated Objects"

Public fire datasets are often assembled from video, so one fire appears as
several near-identical frames. A random train/val/test split scatters those
frames across subsets, and the model is then scored on scenes it has already
seen during training, which inflates every reported metric.

This script quantifies that. It computes a 64-bit dHash for every image and
reports how many test images have a near-duplicate in train. Run it on both
partitions to reproduce the comparison in Section 4.2 of the paper: the random
partition leaks, the group-aware partition does not.

    python check_leakage.py --data data/fire
    python check_leakage.py --data data/fire_random --save-figure leakage.png
    python check_leakage.py --data data/fire --threshold 4 --examples 20

`--data` points at a dataset root laid out by prepare_dataset.py:

    <root>/images/train|val|test/
    <root>/annotations/instances_train|val|test.json

Exit status is 0 when no train/test pair is found and 1 otherwise, so the check
can be used in a script.

This script is standalone: it imports nothing from this repository and needs
only numpy and opencv-python. matplotlib is required only for --save-figure.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

SUBSETS = ("train", "val", "test")


def dhash(path: Path, size: int = 8) -> np.ndarray | None:
    """64-bit difference hash: compare each pixel with its right-hand neighbour
    in a 9x8 greyscale thumbnail. Same definition as in prepare_dataset.py."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def main() -> None:
    ap = argparse.ArgumentParser(description="Check dataset splits for leakage")
    ap.add_argument("--data", default="data/fire",
                    help="dataset root written by prepare_dataset.py")
    ap.add_argument("--threshold", type=int, default=6,
                    help="dHash Hamming distance counting as a near-duplicate (0-64)")
    ap.add_argument("--examples", type=int, default=10, help="example pairs to print")
    ap.add_argument("--save-figure", default=None, help="render example pairs to this PNG")
    args = ap.parse_args()

    root = Path(args.data)
    if not root.is_dir():
        raise SystemExit(f"Dataset root not found: {root}\n"
                         "Pass --data <root>, or run prepare_dataset.py first.")

    records: List[Tuple[str, str, Path, np.ndarray]] = []
    for subset in SUBSETS:
        ann_path = root / "annotations" / f"instances_{subset}.json"
        img_dir = root / "images" / subset
        if not ann_path.is_file():
            print(f"[warn] missing {ann_path}")
            continue
        with open(ann_path, "r", encoding="utf-8") as fh:
            ann = json.load(fh)
        for im in ann["images"]:
            path = img_dir / im["file_name"]
            h = dhash(path)
            if h is not None:
                records.append((subset, im["file_name"], path, h))

    if len(records) < 2:
        raise SystemExit("Not enough images to compare. Run prepare_dataset.py first.")

    counts = {s: sum(1 for r in records if r[0] == s) for s in SUBSETS}
    print(f"Dataset: {root}")
    print(f"Images hashed: {len(records)}   {counts}")
    print(f"Near-duplicate threshold: dHash distance <= {args.threshold} of 64 bits\n")

    H = np.array([r[3] for r in records], dtype=np.int8)
    subs = np.array([r[0] for r in records])
    names = np.array([r[1] for r in records])

    # Map bits to -1/+1 so one matrix product gives every pairwise agreement
    # count, from which the Hamming distance follows directly.
    X = (H * 2 - 1).astype(np.float32)
    sim = X @ X.T
    np.fill_diagonal(sim, -999)
    dist = (X.shape[1] - sim) / 2

    pairs = [(int(i), int(j)) for i, j in np.argwhere(dist <= args.threshold) if i < j]
    tt = [(i, j) for i, j in pairs if {subs[i], subs[j]} == {"train", "test"}]
    tv = [(i, j) for i, j in pairs if {subs[i], subs[j]} == {"train", "val"}]

    n_test = counts["test"]
    leaked = {names[j] if subs[j] == "test" else names[i] for i, j in tt}
    pct = 100.0 * len(leaked) / max(1, n_test)

    print(f"near-duplicate pairs, all subsets : {len(pairs)}")
    print(f"  train <-> test                  : {len(tt)}")
    print(f"  train <-> val                   : {len(tv)}")
    print(f"\ntest images with a train near-duplicate: {len(leaked)} / {n_test} ({pct:.1f}%)")

    if tt:
        print(f"\nexample leaking pairs (showing {min(args.examples, len(tt))}):")
        for i, j in tt[:args.examples]:
            a, b = (i, j) if subs[i] == "train" else (j, i)
            print(f"  train {names[a]:20s} <-> test {names[b]:20s}  distance {int(dist[i, j])}")

    if args.save_figure and tt:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        show = tt[:6]
        fig, axes = plt.subplots(len(show), 2, figsize=(8, 3.0 * len(show)), squeeze=False)
        for r, (i, j) in enumerate(show):
            a, b = (i, j) if subs[i] == "train" else (j, i)
            for c, k in enumerate((a, b)):
                img = cv2.cvtColor(cv2.imread(str(records[k][2])), cv2.COLOR_BGR2RGB)
                axes[r][c].imshow(img)
                axes[r][c].axis("off")
                axes[r][c].set_title(f"{subs[k].upper()}: {names[k]}", fontsize=9)
        fig.suptitle("Near-duplicate images shared between splits")
        fig.tight_layout()
        fig.savefig(args.save_figure, dpi=150, bbox_inches="tight")
        print(f"\nfigure -> {args.save_figure}")

    print("\n" + "=" * 66)
    if not tt:
        print("PASS -- no train/test near-duplicate found.")
    elif pct < 1:
        print(f"BORDERLINE -- {pct:.1f}% of test images have a near-duplicate in train.")
    else:
        print(f"FAIL -- {pct:.1f}% of test images have a near-duplicate in train.")
        print("Accuracy measured on this partition is inflated.")
        print("Rebuild with the group-aware partition, which is the default:")
        print("    python prepare_dataset.py --coco-json <path>/_annotations.coco.json")
        print("then retrain and re-evaluate.")
    print("=" * 66)
    sys.exit(0 if not tt else 1)


if __name__ == "__main__":
    main()
