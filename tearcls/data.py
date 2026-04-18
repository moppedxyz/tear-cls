"""Split-aware dataset + DataLoader builder for the AFM tear-film corpus.

Reads `data/processed/index.csv` (written by tearcls.augmentation) — so each
row points at an already-processed PNG on disk (cropped + optionally
augmented). No runtime augmentation is applied here: the train split
contains both the cropped originals and the pre-dumped augmented variants;
val/test contain only the cropped originals.

Pass ``processed_dir`` to point at a different dump for ablation runs
(e.g. ``data/processed_no_cmap`` vs the default).
"""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path

from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PROCESSED_DIR = REPO_ROOT / "data" / "processed"
CLASSES = ["diabetes", "glaucoma", "multiple_sclerosis", "dry_eye", "healthy"]
LABEL_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


def _load_split_rows(processed_dir: Path, split: str) -> list[dict]:
    index_csv = processed_dir / "index.csv"
    if not index_csv.exists():
        raise FileNotFoundError(
            f"{index_csv} not found. Run `python tearcls/augmentation.py` "
            f"(or setup.sh) to generate the processed dataset first."
        )
    with index_csv.open() as f:
        return [r for r in csv.DictReader(f) if r["split"] == split]


class TearDataset(Dataset):
    def __init__(
        self,
        split: str,
        processed_dir: Path = DEFAULT_PROCESSED_DIR,
        load_augmented: bool = True,
    ):
        assert split in {"train", "val", "test"}
        self.split = split
        self.processed_dir = processed_dir
        self.load_augmented = load_augmented
        rows = _load_split_rows(processed_dir, split)
        if not load_augmented:
            rows = [r for r in rows if r["variant"] == "original"]
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        row = self.rows[i]
        img = Image.open(self.processed_dir / row["filepath"]).convert("RGB")
        return {
            "image": img,
            "label": row["label"],
            "label_idx": LABEL_TO_IDX[row["label"]],
            "filepath": row["filepath"],
            "patient_code": row["patient_code"],
            "variant": row["variant"],
        }


def class_balanced_sampler(ds: TearDataset) -> WeightedRandomSampler:
    """Upweights minority classes so each batch sees them proportionally."""
    labels = [r["label"] for r in ds.rows]
    counts = Counter(labels)
    n = len(labels)
    weights = [n / (len(counts) * counts[l]) for l in labels]
    return WeightedRandomSampler(weights, num_samples=n, replacement=True)


def build_loaders(
    batch_size: int,
    num_workers: int = 2,
    collate_fn=None,
    processed_dir: Path = DEFAULT_PROCESSED_DIR,
    load_augmented: bool = True,
):
    """Returns (train_dl, val_dl, test_dl). Train uses class-balanced
    sampling; val/test are deterministic and unshuffled. Pass
    ``processed_dir`` to switch between ablation dumps. Pass
    ``load_augmented=False`` to train on originals only (val/test are
    unaffected — they never contain augmented variants)."""
    train_ds = TearDataset("train", processed_dir, load_augmented=load_augmented)
    val_ds = TearDataset("val", processed_dir, load_augmented=load_augmented)
    test_ds = TearDataset("test", processed_dir, load_augmented=load_augmented)
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
