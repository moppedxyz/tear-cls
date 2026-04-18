"""AFM-aware augmentations for tear-film images.

AFM BMPs are colormapped height maps. Previously we kept augmentation
geometric-only to avoid breaking the height<->color mapping. We now
also allow MILD photometric and scanner-realistic ops because real
AFM/SEM sessions vary in contrast, focus, and scan-line noise — the
model should be robust to that. All photometric ops are tightly
bounded so fine dendrite edges stay visible.

At low p we also re-render the image under a different perceptually-
uniform colormap (viridis/plasma/inferno/magma/cividis). Luminance is
a monotone proxy for height under Bruker's LUT, so the height ordering
is preserved — only the palette changes. This discourages the model
from latching onto palette-specific hue priors inherited from natural-
image pretraining.

Public API: train_augment(PIL) -> PIL. Do NOT apply at eval time.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import albumentations as A
import cv2
import numpy as np
from PIL import Image

# NanoScope BMP exports are 704x575 with a scale-bar strip at the bottom
# and white margins left/right. The pure AFM data region is rows 0..530,
# cols 93..615 (≈523x531, near-square). Applied before augmentation AND
# at eval time to strip annotation text and margins.
AFM_DATA_BOX = (93, 0, 616, 531)  # (left, top, right, bottom)


def crop_afm_data(img: Image.Image) -> Image.Image:
    """Crop to the pure AFM data region, dropping scale bar + side margins."""
    return img.crop(AFM_DATA_BOX)


def _scan_line_artifact(image: np.ndarray, **_) -> np.ndarray:
    """Inject 1–4 horizontal streaks mimicking AFM scanner glitches:
    row bands 1–3 px tall with a small intensity offset (±10/255)."""
    out = image.copy()
    H = out.shape[0]
    n_bands = np.random.randint(1, 5)
    for _i in range(n_bands):
        y = np.random.randint(0, H)
        h = np.random.randint(1, 4)
        delta = np.random.randint(-10, 11)
        y0, y1 = y, min(H, y + h)
        out[y0:y1] = np.clip(out[y0:y1].astype(np.int16) + delta, 0, 255).astype(np.uint8)
    return out


_CMAPS = [
    cv2.COLORMAP_VIRIDIS,
    cv2.COLORMAP_PLASMA,
    cv2.COLORMAP_INFERNO,
    cv2.COLORMAP_MAGMA,
    cv2.COLORMAP_CIVIDIS,
]


def _colormap_remap(image: np.ndarray, **_) -> np.ndarray:
    """Re-render under a different perceptually-uniform colormap, or
    as pure grayscale (luminance only, palette hue stripped). Luminance
    is a monotone proxy for AFM height under Bruker's LUT, so the height
    ordering is preserved — only the palette changes."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    idx = np.random.randint(len(_CMAPS) + 1)
    if idx == len(_CMAPS):
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
    out_bgr = cv2.applyColorMap(gray, _CMAPS[idx])
    return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)


# Each aug is tagged so ablation studies can enable/disable them by name.
# Order matters: geometric first, then colormap remap (so later photometric
# ops compose on the remapped palette), then the rest of the photometric /
# scanner-realistic block. Bounds on photometric ops are kept low so fine
# dendrite edges survive.
_AUGS: list[tuple[str, A.BasicTransform]] = [
    # Geometric — strongly recommended
    ("rotate", A.Rotate(limit=180, border_mode=cv2.BORDER_REFLECT_101, p=1.0)),
    ("hflip", A.HorizontalFlip(p=0.5)),
    ("vflip", A.VerticalFlip(p=0.5)),
    ("affine", A.Affine(
        translate_percent=(-0.08, 0.08),
        scale=(1.0, 1.0),
        rotate=0,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.5,
    )),
    ("rrcrop", A.RandomResizedCrop(
        size=(384, 384),
        scale=(0.75, 1.0),
        ratio=(0.9, 1.1),
        p=0.5,
    )),
    ("elastic", A.ElasticTransform(
        alpha=30,
        sigma=5,
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.3,
    )),
    # Photometric / scanner-realistic — tuned carefully
    ("colormap", A.Lambda(image=_colormap_remap, p=0.25)),
    ("brightness_contrast", A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.4)),
    ("gamma", A.RandomGamma(gamma_limit=(85, 115), p=0.3)),
    ("noise", A.GaussNoise(std_range=(0.02, 0.08), mean_range=(0.0, 0.0), p=0.3)),
    ("scanline", A.Lambda(image=_scan_line_artifact, p=0.25)),
    ("blur", A.GaussianBlur(blur_limit=(3, 5), p=0.2)),
    ("clahe", A.CLAHE(clip_limit=(1.0, 2.0), tile_grid_size=(8, 8), p=0.2)),
]

AUG_NAMES: tuple[str, ...] = tuple(n for n, _ in _AUGS)


def build_train_augment(
    enable: Iterable[str] | None = None,
    disable: Iterable[str] | None = None,
) -> Callable[[Image.Image], Image.Image]:
    """Build a training-augmentation callable with a named subset of augs.

    Pass ``enable`` for a strict allowlist, ``disable`` for a denylist, or
    neither for the full pipeline (default: all augs on). Names come from
    ``AUG_NAMES``. Unknown names raise ``ValueError``. The AFM-data crop is
    always applied — it's a preprocessing step, not an aug.
    """
    if enable is not None and disable is not None:
        raise ValueError("pass enable or disable, not both")
    known = set(AUG_NAMES)
    if enable is not None:
        selected = set(enable)
        unknown = selected - known
    else:
        denied = set(disable or ())
        unknown = denied - known
        selected = known - denied
    if unknown:
        raise ValueError(f"unknown aug names: {sorted(unknown)}; known: {AUG_NAMES}")

    pipeline = A.Compose([t for n, t in _AUGS if n in selected])

    def _augment(img: Image.Image) -> Image.Image:
        img = crop_afm_data(img)
        arr = np.array(img.convert("RGB"))
        out = pipeline(image=arr)["image"]
        return Image.fromarray(out)

    return _augment


train_augment = build_train_augment()


def _main() -> None:
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(
        description="Run the full augmentation pipeline on one image and dump samples.",
    )
    ap.add_argument("image", type=Path, help="Path to input image (e.g. a .bmp)")
    ap.add_argument("--n", type=int, default=10, help="Number of augmented samples to write")
    ap.add_argument("--out", type=Path, default=Path("test"), help="Output directory")
    ap.add_argument(
        "--enable",
        type=str,
        default=None,
        help=f"Comma-separated aug names to enable (default: all). Known: {','.join(AUG_NAMES)}",
    )
    ap.add_argument(
        "--disable",
        type=str,
        default=None,
        help="Comma-separated aug names to disable (mutually exclusive with --enable)",
    )
    args = ap.parse_args()

    enable = [s.strip() for s in args.enable.split(",")] if args.enable else None
    disable = [s.strip() for s in args.disable.split(",")] if args.disable else None
    aug = build_train_augment(enable=enable, disable=disable)

    img = Image.open(args.image).convert("RGB")
    args.out.mkdir(parents=True, exist_ok=True)
    stem = args.image.stem
    img.save(args.out / f"{stem}_orig.png")
    crop_afm_data(img).save(args.out / f"{stem}_cropped.png")
    for i in range(args.n):
        aug(img).save(args.out / f"{stem}_aug_{i:02d}.png")
    print(f"Wrote {args.n} augmented samples + original + cropped to {args.out}/")


if __name__ == "__main__":
    _main()
