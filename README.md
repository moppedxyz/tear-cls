# tear-cls

## Setup

```bash
bash setup.sh
```

This activates `.venv`, runs `uv sync --all-extras`, downloads `data.zip`, unzips to `data_raw/`, then runs `tearcls/data_split.py` and `tearcls/augmentation.py` to produce `data/processed/` with `index.csv`.

Reuse an existing `data.zip` instead of re-downloading:

```bash
bash setup.sh --skip-download
```

## Loading data

```python
from tearcls.data import build_loaders

train_dl, val_dl, test_dl = build_loaders(batch_size=16, num_workers=2)

for batch in train_dl:
    batch["image"], batch["label_idx"], batch["label"]
```

Train uses a class-balanced sampler and includes pre-dumped augmented variants; val/test are deterministic and originals-only. Pass `load_augmented=False` to train on originals only, or `processed_dir=Path("data/processed_no_cmap")` to switch ablation dumps.
