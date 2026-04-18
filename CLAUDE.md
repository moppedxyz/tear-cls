# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Fine-tune a vision LLM on AFM (atomic-force microscopy) tear-film images to classify into 5 diagnoses:
`diabetes`, `glaucoma`, `multiple_sclerosis`, `dry_eye`, `healthy`. Source folders are Slovak
(`Diabetes`, `PGOV_Glaukom`, `SklerózaMultiplex`, `SucheOko`, `ZdraviLudia`) and are remapped via
`CLASS_MAP` in `tearcls/data_split.py` — keep that mapping as the single source of truth for
class naming.

Python 3.12 only. Torch is pinned to the `cu121` wheel index (works with CUDA 13 driver via
backward compat) — do not switch indexes without updating `pyproject.toml` `[tool.uv.sources]`.

## Commands

Environment + data (full cold start):
```
./setup.sh                  # uv sync, download data.zip, unzip to data_raw/, split, augment
./setup.sh --skip-download  # reuse an existing data.zip
```

Dependency-only:
```
uv sync --all-extras
```

Rebuild splits / offline augment dumps independently:
```
python tearcls/data_split.py                          # writes data/splits.csv
python tearcls/augmentation.py --n 10                 # writes data/processed/**/*.png + index.csv
python tearcls/augmentation.py --enable rotate,hflip  # ablation: enable a subset
python tearcls/augmentation.py --disable colormap     # ablation: denylist
```

Preview augmentations on one image without a full dump:
```
python tearcls/augment.py path/to.bmp --n 10 --out test/
```

Smoke-test the data pipeline (prints batch shapes — `train.py` is not a real trainer yet):
```
python train.py
```

## Architecture

Two-stage data pipeline, both stages produce artifacts under `data/` that downstream code reads
from disk (no in-memory pipe between them):

1. **Split** (`tearcls/data_split.py`) scans `data_raw/<ClassFolder>/*.bmp`, extracts a
   **patient code** from each filename via `parse_patient_code` (cuts on the first `.NNN`
   numeric segment — handles IDs with embedded dates like `DM_01.03.2024_LO.001_1`), then does
   a **per-class group split**: each class's unique patients are allocated 80/10/10 to
   train/val/test. This guarantees every class appears in every split AND no patient crosses
   splits. Classes with <3 unique patients fall back to image-level split with a `WARN` log —
   check the summary output when adding data. Output: `data/splits.csv`.

2. **Augment + crop** (`tearcls/augmentation.py`) reads `splits.csv`, writes originals
   (cropped to the AFM data region) for every split and N augmented variants per train image
   to `data/processed/<ClassFolder>/*.png`, with a consolidated `data/processed/index.csv`.

The split-aware loader (`tearcls/data.py`) currently reads `data/splits.csv` directly (not
`data/processed/index.csv`) and applies augs on-the-fly — `train_augment` for train,
`crop_afm_data` only for val/test. Train uses `class_balanced_sampler`
(`WeightedRandomSampler`, `replacement=True`) so minority classes get upsampled each epoch.
Val/test are unshuffled and deterministic.

### AFM-specific invariants

- `AFM_DATA_BOX = (93, 0, 616, 531)` in `tearcls/augment.py` crops out NanoScope's scale-bar
  strip + side margins. It's **preprocessing, always applied** (train AND eval) — not part of
  the aug ablation set. Don't move it into the `_AUGS` list.
- AFM BMPs are colormapped height maps under Bruker's LUT. Luminance is a monotone proxy for
  height, so the `colormap` aug re-renders under viridis/plasma/inferno/magma/cividis/gray —
  this preserves height ordering while breaking palette hue priors inherited from natural-image
  pretraining. Photometric bounds are kept **tight** (brightness/contrast ±0.15, gamma 85–115,
  mild noise/blur) so fine dendrite edges survive. Treat any aug expansion as an AFM-semantics
  decision, not a generic CV tuning knob.
- `_scan_line_artifact` injects 1–4 horizontal streaks mimicking real AFM scanner glitches.

### Augmentation ablation API

`build_train_augment(enable=..., disable=...)` accepts named subsets of `AUG_NAMES`. `enable`
and `disable` are mutually exclusive; unknown names raise `ValueError`. The module-level
`train_augment` is the full pipeline (all augs on). `tearcls/augmentation.py` and
`tearcls/augment.py` both forward `--enable`/`--disable` flags, so ablations can be run end-to-end
from the CLI without editing code.

### Labels

`CLASSES` and `LABEL_TO_IDX` are duplicated in `tearcls/data.py` and `tearcls/evaluation.py`.
If you reorder, update both — index order is load-bearing for `f1_score(..., labels=CLASSES)`.
