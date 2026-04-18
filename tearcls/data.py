"""Split-aware dataset + DataLoader builder for the AFM tear-film corpus.

Reads `data/splits.csv` (written by tearcls.data_split), loads each BMP as
PIL RGB, and applies split-appropriate transforms:
  - train: train_augment (crop + geometric + mild photometric + colormap)
  - val/test: crop_afm_data only — deterministic, no randomness
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from tearcls.augment import crop_afm_data, train_augment

REPO_ROOT = Path(__file__).resolve().parent.parent
SPLITS_CSV = REPO_ROOT / "data" / "splits.csv"
CLASSES = ["diabetes", "glaucoma", "multiple_sclerosis", "dry_eye", "healthy"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def _load_split_rows(split: str) -> list[dict]:
    with SPLITS_CSV.open() as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


class TearDataset(Dataset):
    def __init__(self, split: str):
        assert split in {"train", "val", "test"}
        self.split = split
        self.rows = _load_split_rows(split)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        img = Image.open(REPO_ROOT / row["filepath"]).convert("RGB")
        img = train_augment(img) if self.split == "train" else crop_afm_data(img)
        return {
            "image": img,
            "label": row["label"],
            "label_idx": LABEL_TO_IDX[row["label"]],
            "filepath": row["filepath"],
            "patient_code": row["patient_code"],
        }


def class_balanced_sampler(ds: TearDataset) -> WeightedRandomSampler:
    """Upweights minority classes so each batch sees them proportionally."""
    labels = [r["label"] for r in ds.rows]
    counts = Counter(labels)
    n = len(labels)
    weights = [n / (len(counts) * counts[l]) for l in labels]
    return WeightedRandomSampler(weights, num_samples=n, replacement=True)


def build_loaders(batch_size: int, num_workers: int = 2, collate_fn=None):
    """Returns (train_dl, val_dl, test_dl). Train uses class-balanced
    sampling; val/test are deterministic and unshuffled."""
    train_ds = TearDataset("train")
    val_ds = TearDataset("val")
    test_ds = TearDataset("test")
    train_dl = DataLoader(
        train_ds,
        batch_size=batch_size,
        sampler=class_balanced_sampler(train_ds),
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return train_dl, val_dl, test_dl
