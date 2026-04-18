"""Offline-augment the train split + copy val/test passthroughs into data/.

Reads data/splits.csv. For each row:
- train split: writes the cropped original + N augmented variants
- val/test splits: writes only the cropped original (no augs)

All output goes under data/processed/<class_folder>/*.png plus a consolidated
data/processed/index.csv (one row per file) so trainers can load from a
single source of truth.

Ablation: pass --enable / --disable to dump different augmentation subsets.
Known aug names: tearcls.augment.AUG_NAMES.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tearcls.augment import AUG_NAMES, build_train_augment, crop_afm_data  # noqa: E402
DEFAULT_SPLITS = REPO_ROOT / "data" / "splits.csv"
DEFAULT_OUT = REPO_ROOT / "data" / "processed"

FIELDNAMES = [
    "filepath",
    "class",
    "label",
    "patient_code",
    "split",
    "source_filepath",
    "variant",
]


def _parse_names(raw: str | None) -> list[str] | None:
    if not raw:
        return None
    return [s.strip() for s in raw.split(",") if s.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--n", type=int, default=10, help="Augmented variants per train image"
    )
    ap.add_argument(
        "--seed", type=int, default=3407, help="Seed for reproducible dumps"
    )
    ap.add_argument(
        "--enable",
        type=str,
        default=None,
        help=f"Comma-separated aug names (known: {','.join(AUG_NAMES)})",
    )
    ap.add_argument(
        "--disable", type=str, default=None, help="Comma-separated aug names to skip"
    )
    ap.add_argument("--splits-csv", type=Path, default=DEFAULT_SPLITS)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    aug = build_train_augment(
        enable=_parse_names(args.enable),
        disable=_parse_names(args.disable),
    )

    with args.splits_csv.open() as f:
        rows = list(csv.DictReader(f))

    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "index.csv"

    n_orig = 0
    n_aug = 0
    with index_path.open("w", newline="") as f_out:
        writer = csv.DictWriter(f_out, fieldnames=FIELDNAMES)
        writer.writeheader()

        for row in tqdm(rows, desc="augmenting", unit="img"):
            src = REPO_ROOT / row["filepath"]
            img = Image.open(src).convert("RGB")
            cls_dir = args.out / row["class"]
            cls_dir.mkdir(parents=True, exist_ok=True)

            orig_path = cls_dir / f"{src.stem}.png"
            crop_afm_data(img).save(orig_path)
            writer.writerow(
                {
                    "filepath": orig_path.relative_to(args.out).as_posix(),
                    "class": row["class"],
                    "label": row["label"],
                    "patient_code": row["patient_code"],
                    "split": row["split"],
                    "source_filepath": row["filepath"],
                    "variant": "original",
                }
            )
            n_orig += 1

            if row["split"] != "train":
                continue
            for i in range(args.n):
                aug_path = cls_dir / f"{src.stem}_aug_{i:03d}.png"
                aug(img).save(aug_path)
                writer.writerow(
                    {
                        "filepath": aug_path.relative_to(args.out).as_posix(),
                        "class": row["class"],
                        "label": row["label"],
                        "patient_code": row["patient_code"],
                        "split": row["split"],
                        "source_filepath": row["filepath"],
                        "variant": str(i),
                    }
                )
                n_aug += 1

    print(f"Wrote {n_orig} originals + {n_aug} augmented = {n_orig + n_aug} files")
    print(f"Index: {index_path}")


if __name__ == "__main__":
    main()
