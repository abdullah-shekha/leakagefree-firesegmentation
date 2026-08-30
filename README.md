# Code implementation for the paper

**Leakage-Free Evaluation of Mask R-CNN for Fire Region Segmentation in Still Images: Accuracy and Failure Modes on Fire-Illuminated Objects**

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

## Licence

MIT. See `LICENSE`.
