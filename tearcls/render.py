"""Lambert-shaded, class-tinted AFM crop renderer.

Given a cleaned 2-D height array (produced by ``tearcls.gwy_prep``), this
module samples a deterministic zoomed crop for a gallery item, shades it with
a soft lambert light (mirroring Gwyddion's default illumination), and composes
the shading over a class-tinted RGB LUT. Output is a PIL ``Image`` ready to
serve.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from tearcls.palettes import class_lut, crop_plan_for


LIGHT_DIR = np.array([1.0, 1.0, 1.7], dtype=np.float32)
LIGHT_DIR /= np.linalg.norm(LIGHT_DIR)


def _percentile_norm(arr: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> np.ndarray:
    p_lo, p_hi = np.percentile(arr, [lo, hi])
    span = float(p_hi - p_lo)
    if span < 1e-9:
        return np.zeros_like(arr, dtype=np.float32)
    out = (arr.astype(np.float32) - float(p_lo)) / span
    return np.clip(out, 0.0, 1.0)


def lambert_shade(height: np.ndarray, exaggerate: float = 2.2) -> np.ndarray:
    """Return shading in [0, 1]. Surface normal from finite differences; light
    direction matches Gwyddion's 'upper-left' default."""
    z = height.astype(np.float32) * exaggerate
    dz_dy, dz_dx = np.gradient(z)
    nx = -dz_dx
    ny = -dz_dy
    nz = np.ones_like(z, dtype=np.float32)
    inv = 1.0 / np.sqrt(nx * nx + ny * ny + nz * nz + 1e-9)
    nx *= inv; ny *= inv; nz *= inv
    dot = nx * LIGHT_DIR[0] + ny * LIGHT_DIR[1] + nz * LIGHT_DIR[2]
    shade = np.clip(dot, 0.0, 1.0)
    return shade


def sample_crop(height: np.ndarray, item_id: str) -> np.ndarray:
    """Return a deterministic zoomed crop from the height array for this id."""
    h, w = height.shape
    plan = crop_plan_for(item_id, h, w)
    t, l, s = plan["top"], plan["left"], plan["size"]
    s = min(s, h - t, w - l)
    return height[t:t + s, l:l + s]


def _height_features(height: np.ndarray, bins: int = 8) -> list[float]:
    """Compact descriptor used to drive the melodic contour: row-mean quantiles."""
    row_means = height.mean(axis=1)
    return [float(v) for v in np.quantile(row_means, np.linspace(0.05, 0.95, bins))]


def height_features(height: np.ndarray) -> list[float]:
    """Public: scalar descriptors of the crop for the audio engine."""
    return _height_features(height)


def compose_card(
    height: np.ndarray,
    item_id: str,
    class_label: str,
    target_size: int = 768,
) -> Image.Image:
    """Full pipeline: zoomed crop -> lambert -> percentile-norm -> class LUT.

    Returns a ``target_size`` × ``target_size`` RGB PIL image. The LUT is
    driven by the *normalized shaded* value, not the raw height, so contrast
    stays high at any zoom.
    """
    crop = sample_crop(height, item_id)
    shade = lambert_shade(crop)

    # Blend: 70% shading luminance, 30% height percentile so large-scale
    # topology still influences color mapping (so the LUT doesn't collapse
    # to a single band when shading is locally flat).
    norm = _percentile_norm(crop)
    luma = np.clip(0.7 * shade + 0.3 * norm, 0.0, 1.0)

    lut = class_lut(class_label, item_id)
    idx = (luma * 255.0).astype(np.uint8)
    rgb = lut[idx]

    img = Image.fromarray(rgb, mode="RGB")
    if img.size != (target_size, target_size):
        img = img.resize((target_size, target_size), Image.BICUBIC)
    return img


def render_heightmap_png(height: np.ndarray, size: int = 256) -> Image.Image:
    """Return a grayscale PIL image of the normalized full height array.

    Used client-side as a displacement texture for the per-card Three.js mesh.
    """
    norm = _percentile_norm(height)
    img = Image.fromarray((norm * 255.0).astype(np.uint8), mode="L")
    if img.size != (size, size):
        img = img.resize((size, size), Image.BICUBIC)
    return img


__all__ = [
    "compose_card",
    "lambert_shade",
    "sample_crop",
    "height_features",
    "render_heightmap_png",
]
