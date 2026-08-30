"""
prepare_dataset.py -- build the leakage-free partition of the Fire Instance
Segmentation Dataset used in:

"Leakage-Free Evaluation of Mask R-CNN for Fire Region Segmentation in Still Images: Accuracy and Failure Modes on Fire-Illuminated Objects"

Input is the public COCO-format release of the dataset (polygon annotations for
two classes, fire and smoke). Only the fire class is kept.

Output layout:

    data/fire/
      images/train|val|test/
      annotations/instances_train|val|test.json
      dataset_stats.json      counts and size distribution (Table I, Figure 1)
      split.json              the exact file list of each subset

The two partitions compared in the paper are produced by the same command,
differing only in the flag on the last line:

    # group-aware partition (leakage-free; the main results)
    python prepare_dataset.py --coco-json <path>/_annotations.coco.json

    # random partition (contaminated; the comparison in Section 4.2)
    python prepare_dataset.py --coco-json <path>/_annotations.coco.json \\
        --out data/fire_random --no-group-split

Run check_leakage.py afterwards to confirm the group-aware partition contains
no cross-subset near-duplicate.

This script is standalone: it imports nothing from this repository and needs
only numpy and opencv-python.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def write_json(obj, path: Path, indent: int = 2) -> None:
    """Write JSON, converting the numpy scalars and arrays this script produces."""
    def default(o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return str(o)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=default)


# ===========================================================================
# Reading the source annotations
# ===========================================================================
def read_coco_json(coco_json: Path, search_root: Path,
                   keep: Optional[Sequence[str]] = None) -> List[dict]:
    """Read a COCO instances json into per-image records.

    `keep` filters by category name; the paper keeps only "fire". Categories
    named like background are always dropped. `file_name` may include a
    subfolder, so it is resolved relative to the json first and then searched
    for in the tree.
    """
    with open(coco_json, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "images" not in data or "annotations" not in data:
        raise SystemExit(f"{coco_json} is not a COCO instances file")

    names = {c["id"]: c["name"] for c in data.get("categories", [])}
    keep_lower = {k.lower() for k in keep} if keep else None
    id2img = {im["id"]: im for im in data["images"]}

    grouped: Dict[int, Dict[str, list]] = {}
    skipped_rle = 0
    for ann in data["annotations"]:
        label = names.get(ann.get("category_id"), "fire")
        if label.strip("_").lower() in ("background", "bg", "none"):
            continue
        if keep_lower is not None and label.lower() not in keep_lower:
            continue
        seg = ann.get("segmentation")
        if isinstance(seg, dict):          # RLE, not polygons
            skipped_rle += 1
            continue
        if not isinstance(seg, list):
            continue
        for poly in seg:
            if isinstance(poly, list) and len(poly) >= 6:
                entry = grouped.setdefault(ann["image_id"], {"polygons": [], "labels": []})
                entry["polygons"].append([float(v) for v in poly])
                entry["labels"].append(label)

    if skipped_rle:
        print(f"  [warn] {skipped_rle} RLE-encoded annotation(s) skipped "
              "(only polygon segmentation is supported)")

    records: List[dict] = []
    missing = 0
    for img_id, entry in grouped.items():
        info = id2img.get(img_id)
        if info is None:
            continue
        path = coco_json.parent / info["file_name"]
        if not path.is_file():
            hits = list(search_root.rglob(Path(info["file_name"]).name))
            if not hits:
                missing += 1
                continue
            path = hits[0]
        records.append({"path": path, "polygons": entry["polygons"],
                        "labels": entry["labels"]})
    if missing:
        print(f"  [warn] {missing} image(s) referenced in {coco_json.name} not found on disk")
    return records


# ===========================================================================
# Near-duplicate detection and group-aware splitting
# ===========================================================================
def perceptual_hash(path: Path, size: int = 8) -> Optional[np.ndarray]:
    """64-bit difference hash: does each pixel exceed its right-hand neighbour?

    Robust to rescaling, re-compression and small crops, which is what
    separates "another frame of the same fire" from "another scene".
    """
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (small[:, 1:] > small[:, :-1]).flatten()


def group_near_duplicates(records: Sequence[dict], threshold: int = 6,
                          chunk: int = 2048) -> List[int]:
    """Assign a group id to each record so near-duplicates share one.

    Public fire datasets are built largely from video, so consecutive frames of
    one fire appear as separate images. Splitting those at random puts a frame
    in train and its neighbour in test, and the model is then scored on scenes
    it has effectively memorised. Merging near-duplicates transitively and
    assigning whole groups removes that leakage.
    """
    hashes, valid = [], []
    for i, rec in enumerate(records):
        h = perceptual_hash(rec["path"])
        if h is not None:
            hashes.append(h)
            valid.append(i)

    groups = list(range(len(records)))
    if len(hashes) < 2:
        return groups

    def find(x):
        while groups[x] != x:
            groups[x] = groups[groups[x]]
            x = groups[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            groups[max(ra, rb)] = min(ra, rb)

    # +/-1 encoding turns Hamming distance into one matrix product:
    # distance = (n_bits - similarity) / 2
    X = (np.array(hashes, dtype=np.int8) * 2 - 1).astype(np.float32)
    n_bits = X.shape[1]
    for start in range(0, len(X), chunk):
        block = X[start:start + chunk]
        dist = (n_bits - block @ X.T) / 2
        for local, row in enumerate(dist):
            i = start + local
            for j in np.where(row <= threshold)[0]:
                if int(j) != i:
                    union(valid[i], valid[int(j)])

    return [find(i) for i in range(len(records))]


def split_records(records: List[dict], ratios: Tuple[float, float, float],
                  seed: int, groups: Optional[Sequence[int]] = None
                  ) -> Dict[str, List[dict]]:
    """Split into train/val/test, keeping grouped records together."""
    rng = random.Random(seed)
    n = len(records)

    if groups is None:
        shuffled = list(records)
        rng.shuffle(shuffled)
        buckets = [[r] for r in shuffled]
    else:
        by_group: Dict[int, List[dict]] = {}
        for rec, g in zip(records, groups):
            by_group.setdefault(g, []).append(rec)
        buckets = list(by_group.values())
        rng.shuffle(buckets)

    targets = {"train": ratios[0] * n, "val": ratios[1] * n, "test": ratios[2] * n}
    out: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}

    # Largest groups first, each to whichever subset is furthest below its
    # quota. Filling sequentially instead lets one oversized group push train
    # past its target and starve test; with heavily grouped data that skews the
    # split badly (an earlier arbitrary assignment left 153 images in test).
    for bucket in sorted(buckets, key=len, reverse=True):
        deficit = {k: targets[k] - len(out[k]) for k in out}
        pick = max(deficit, key=lambda k: (deficit[k], targets[k]))
        out[pick].extend(bucket)

    for target in ("val", "test"):
        if not out[target] and len(out["train"]) > 2:
            out[target].append(out["train"].pop())
    return out


# ===========================================================================
# COCO assembly
# ===========================================================================
def collect_class_names(records: Sequence[dict]) -> List[str]:
    """Every distinct instance label across the records, 'fire' first."""
    names = {lab for rec in records for lab in rec.get("labels") or []}
    if not names:
        return ["fire"]
    ordered = sorted(names)
    if "fire" in ordered:
        ordered.remove("fire")
        ordered.insert(0, "fire")
    return ordered


def build_coco(records: Sequence[dict], class_names: Sequence[str],
               min_area: int = 60, copy_to: Optional[Path] = None,
               max_image_side: int = 1600) -> dict:
    """Turn records into a COCO dict, copying the images into `copy_to`.

    `max_image_side` downscales oversized source images and their annotations.
    This is not cosmetic: a 5318x5972 photo becomes a 363 MB float32 tensor the
    moment torchvision touches it, and the augmentation pipeline holds several
    such copies at once, which is enough to exhaust host RAM mid-epoch. The
    model resizes everything to its own min_size anyway.
    """
    name_to_id = {name: i for i, name in enumerate(class_names, start=1)}
    coco = {
        "info": {"description": "Fire instance-segmentation dataset",
                 "version": "1.0", "contributor": "prepare_dataset.py"},
        "licenses": [],
        "images": [],
        "annotations": [],
        "categories": [{"id": i, "name": n, "supercategory": "fire"}
                       for n, i in name_to_id.items()],
    }

    ann_id = 1
    n_resized = 0
    for img_id, rec in enumerate(records, start=1):
        img = cv2.imread(str(rec["path"]))
        if img is None:
            print(f"  [warn] unreadable image skipped: {rec['path']}")
            continue
        h, w = img.shape[:2]

        resized = False
        if max_image_side and max(h, w) > max_image_side:
            scale = max_image_side / float(max(h, w))
            new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rec = dict(rec)
            rec["polygons"] = [[v * scale for v in poly] for poly in rec["polygons"]]
            h, w = new_h, new_w
            resized = True
            n_resized += 1

        anns = []
        labels = rec.get("labels") or [class_names[0]] * len(rec["polygons"])
        for poly_idx, poly in enumerate(rec["polygons"]):
            xs = np.clip(np.array(poly[0::2], dtype=np.float32), 0, w - 1)
            ys = np.clip(np.array(poly[1::2], dtype=np.float32), 0, h - 1)
            x0, y0 = float(xs.min()), float(ys.min())
            bw, bh = float(xs.max()) - x0, float(ys.max()) - y0
            if bw < 2 or bh < 2:
                continue
            # shoelace formula
            area = float(0.5 * abs(np.dot(xs, np.roll(ys, 1)) - np.dot(ys, np.roll(xs, 1))))
            if area < min_area:
                continue
            flat: List[float] = []
            for x, y in zip(xs.tolist(), ys.tolist()):
                flat.extend([x, y])
            label = labels[poly_idx] if poly_idx < len(labels) else class_names[0]
            anns.append({"segmentation": [flat], "area": area,
                         "bbox": [x0, y0, bw, bh],
                         "category_id": name_to_id.get(label, 1)})

        if not anns:
            continue  # images with no fire are dropped: Mask R-CNN needs targets

        file_name = rec["path"].name
        if copy_to is not None:
            copy_to.mkdir(parents=True, exist_ok=True)
            target = copy_to / file_name
            n = 1
            while target.exists() and target.resolve() != rec["path"].resolve():
                target = copy_to / f"{rec['path'].stem}_{n}{rec['path'].suffix}"
                n += 1
            if resized:
                cv2.imwrite(str(target), img)
            else:
                shutil.copy2(rec["path"], target)
            file_name = target.name

        coco["images"].append({"id": img_id, "file_name": file_name,
                               "width": w, "height": h,
                               "source_path": str(rec["path"])})
        for a in anns:
            a.update({"id": ann_id, "image_id": img_id, "iscrowd": 0})
            coco["annotations"].append(a)
            ann_id += 1

    if n_resized:
        print(f"    downscaled {n_resized} oversized image(s) to max side {max_image_side}px")
    return coco


def dataset_statistics(coco: dict) -> dict:
    """Counts and the size distribution reported in Table I and Figure 1."""
    areas = np.array([a["area"] for a in coco["annotations"]], dtype=np.float64)
    per_image = Counter(a["image_id"] for a in coco["annotations"])

    area_by_image: Dict[int, float] = {}
    for a in coco["annotations"]:
        area_by_image[a["image_id"]] = area_by_image.get(a["image_id"], 0.0) + a["area"]
    coverage = np.array([area_by_image.get(im["id"], 0.0) / (im["width"] * im["height"])
                         for im in coco["images"]]) if coco["images"] else np.array([0.0])

    return {
        "images": len(coco["images"]),
        "instances": len(coco["annotations"]),
        "instances_per_image_mean": float(np.mean(list(per_image.values()))) if per_image else 0.0,
        "instances_per_image_max": int(max(per_image.values())) if per_image else 0,
        "instance_area_px_mean": float(areas.mean()) if areas.size else 0.0,
        "instance_area_px_median": float(np.median(areas)) if areas.size else 0.0,
        "fire_pixel_coverage_mean": float(coverage.mean()),
        "fire_pixel_coverage_std": float(coverage.std()),
        # COCO area convention
        "size_small": int((areas < 32 ** 2).sum()),
        "size_medium": int(((areas >= 32 ** 2) & (areas < 96 ** 2)).sum()),
        "size_large": int((areas >= 96 ** 2).sum()),
        "image_width_mean": float(np.mean([im["width"] for im in coco["images"]])) if coco["images"] else 0,
        "image_height_mean": float(np.mean([im["height"] for im in coco["images"]])) if coco["images"] else 0,
    }


def verify(out_root: Path, subsets: Sequence[str]) -> None:
    """Confirm every image on disk matches the dimensions in its annotation.

    A stale image left from a previous build with fresh json metadata is a
    silent failure: the annotations describe one resolution, the file another.
    """
    mismatches = 0
    for subset in subsets:
        ann_path = out_root / "annotations" / f"instances_{subset}.json"
        if not ann_path.is_file():
            continue
        with open(ann_path, "r", encoding="utf-8") as fh:
            coco = json.load(fh)
        for im in coco["images"]:
            f = out_root / "images" / subset / im["file_name"]
            if not f.is_file():
                print(f"  [ERROR] missing image on disk: {f}")
                mismatches += 1
                continue
            probe = cv2.imread(str(f))
            if probe is None:
                print(f"  [ERROR] unreadable: {f}")
                mismatches += 1
            elif (probe.shape[1], probe.shape[0]) != (im["width"], im["height"]):
                print(f"  [ERROR] size mismatch {f.name}: json {im['width']}x{im['height']} "
                      f"vs file {probe.shape[1]}x{probe.shape[0]}")
                mismatches += 1
    if mismatches:
        raise SystemExit(f"\n{mismatches} image(s) disagree with their annotations. "
                         f"Delete {out_root} and run this script again.")
    print("  verified: every image on disk matches its annotation record")


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--coco-json", required=True,
                    help="COCO instances json of the source dataset")
    ap.add_argument("--images", default=None,
                    help="root to search for image files (default: the json's folder)")
    ap.add_argument("--out", default="data/fire", help="output dataset root")
    ap.add_argument("--categories", nargs="*", default=["fire"], metavar="NAME",
                    help="annotation classes to keep (default: fire)")
    ap.add_argument("--split", type=float, nargs=3, default=[0.7, 0.15, 0.15],
                    metavar=("TRAIN", "VAL", "TEST"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-group-split", action="store_true",
                    help="split at random instead of keeping near-duplicate images "
                         "in the same subset. This reproduces the contaminated "
                         "partition of Section 4.2; it inflates every reported metric "
                         "and must not be used for headline results.")
    ap.add_argument("--dup-threshold", type=int, default=6,
                    help="dHash Hamming distance below which two images count as "
                         "near-duplicates (0-64, default 6)")
    ap.add_argument("--max-image-side", type=int, default=1600,
                    help="downscale any image whose longest side exceeds this, "
                         "rescaling its annotations too (0 disables)")
    ap.add_argument("--min-area", type=int, default=60,
                    help="drop instances smaller than this many pixels")
    args = ap.parse_args()

    out_root = Path(args.out)
    coco_json = Path(args.coco_json)
    if not coco_json.is_file():
        raise SystemExit(f"Not found: {coco_json}")
    search_root = Path(args.images) if args.images else coco_json.parent

    # ---------------- load ---------------------------------------------------
    print(f"[1/4] Reading {coco_json.name}")
    records = read_coco_json(coco_json, search_root, keep=args.categories)
    if not records:
        raise SystemExit("No annotated images were found. Check --coco-json and --images.")
    n_inst = sum(len(r["polygons"]) for r in records)
    print(f"  -> {len(records)} image(s), {n_inst} instance(s), "
          f"classes kept: {args.categories}")

    # ---------------- split --------------------------------------------------
    print(f"[2/4] Splitting {args.split[0]:.0%}/{args.split[1]:.0%}/{args.split[2]:.0%} "
          f"(seed={args.seed})")
    groups = None
    if args.no_group_split:
        print("  [warn] --no-group-split: near-duplicates may span subsets")
    else:
        print("  scanning for near-duplicate images ...")
        groups = group_near_duplicates(records, threshold=args.dup_threshold)
        sizes = Counter(groups)
        n_groups = len(sizes)
        in_multi = sum(n for n in sizes.values() if n > 1)
        largest = max(sizes.values()) if sizes else 0
        print(f"  {len(records)} images -> {n_groups} scene group(s)")
        if in_multi:
            print(f"  {in_multi} of {len(records)} image(s) share a scene with at least "
                  f"one other; largest group holds {largest}")
            print("  whole groups are assigned to a subset, never individual images")
    splits = split_records(records, tuple(args.split), args.seed, groups=groups)
    for k, v in splits.items():
        print(f"  {k:5s}: {len(v)} images")

    # ---------------- build --------------------------------------------------
    print("[3/4] Writing COCO annotations")
    for stale in (out_root / "images", out_root / "annotations"):
        if stale.exists():
            print(f"  removing previous build: {stale}")
            shutil.rmtree(stale)
    (out_root / "annotations").mkdir(parents=True, exist_ok=True)

    class_names = collect_class_names(records)
    print(f"  classes: {class_names}")
    stats, split_manifest = {}, {}
    for subset, recs in splits.items():
        if not recs:
            print(f"  [warn] subset '{subset}' is empty")
        coco = build_coco(recs, class_names, min_area=args.min_area,
                          copy_to=out_root / "images" / subset,
                          max_image_side=args.max_image_side)
        write_json(coco, out_root / "annotations" / f"instances_{subset}.json")
        stats[subset] = dataset_statistics(coco)
        split_manifest[subset] = [im["file_name"] for im in coco["images"]]
        print(f"  {subset:5s}: {stats[subset]['images']} images, "
              f"{stats[subset]['instances']} fire instances")

    # ---------------- verify -------------------------------------------------
    print("[4/4] Verifying and writing statistics")
    verify(out_root, list(splits))

    totals = {"images": sum(s["images"] for s in stats.values()),
              "instances": sum(s["instances"] for s in stats.values())}
    write_json({"source": str(coco_json), "seed": args.seed,
                "split_ratios": args.split, "group_aware": not args.no_group_split,
                "dup_threshold": args.dup_threshold, "min_area": args.min_area,
                "max_image_side": args.max_image_side,
                "per_subset": stats, "total": totals},
               out_root / "dataset_stats.json")
    write_json(split_manifest, out_root / "split.json")

    print("\n" + "=" * 62)
    print(f"Dataset ready at: {out_root.resolve()}")
    print(f"  total images   : {totals['images']}")
    print(f"  total instances: {totals['instances']}")
    print("=" * 62)
    print("\nNext step, to confirm the partition contains no cross-subset "
          "near-duplicate:\n  python check_leakage.py --data " + str(out_root))


if __name__ == "__main__":
    main()
