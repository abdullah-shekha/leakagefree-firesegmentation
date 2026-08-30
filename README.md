# Leakage-Free Data Partitioning for Instance Segmentation

Near-duplicate grouping and group-aware train/validation/test splitting, with an
audit that measures how much leakage a partition contains.

> [!NOTE]
> **Work in progress.** This repository currently contains the data-partitioning
> method described in the paper. That part is complete and runs on its own. The
> full training and evaluation code will be released here upon publication of
> the article.

## What is in this repository

| File | Role |
|---|---|
| `prepare_dataset.py` | Builds the partition. Hashing, near-duplicate grouping, group-aware splitting, COCO output, statistics and a verification pass. |
| `check_leakage.py` | Audits a partition for cross-subset near-duplicates. Exit status 0 = clean, 1 = leaking. |
| `split.json` | The partition used in the study: the exact list of images in each subset (1,515 train / 325 validation / 323 test). |

Both scripts are standalone and need only:

```bash
pip install numpy opencv-python
```

## Usage

```bash
# build the leakage-free partition
python prepare_dataset.py --coco-json <path>/_annotations.coco.json --out data/fire

# audit it
python check_leakage.py --data data/fire
```

Run either script with `--help` for the full list of options.

## The published partition

`split.json` records which image went into which subset, so the partition can be
inspected, or reused exactly, without re-running the hashing. Rebuilding it from
the same source release with the default settings (seed 42, dHash threshold 6)
should reproduce the same assignment.

## Licence

MIT. See [`LICENSE`](LICENSE).
