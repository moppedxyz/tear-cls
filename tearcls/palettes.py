"""Per-class palettes and deterministic per-item visual + audio parameters.

Every card in the gallery gets:
- A class-tinted RGB LUT (hue family fixed per class, shifted per item).
- Audio parameters picked from a large acoustic-instrument parameter space.
- 3D shape parameters (primitive family fixed per class, deformations per item).

All three are derived from ``sha1(item_id)`` so the same card always renders
identically across reloads, but every card sounds and looks obviously distinct.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np

CLASSES = ("diabetes", "glaucoma", "multiple_sclerosis", "dry_eye", "healthy")

CLASS_PALETTES: dict[str, dict[str, float | tuple[int, int, int]]] = {
    "diabetes":           {"hue": 355.0, "sat": 0.78, "accent_hue": 18.0},
    "glaucoma":           {"hue": 278.0, "sat": 0.68, "accent_hue": 310.0},
    "multiple_sclerosis": {"hue": 185.0, "sat": 0.72, "accent_hue": 218.0},
    "dry_eye":            {"hue":  38.0, "sat": 0.82, "accent_hue":  12.0},
    "healthy":            {"hue": 145.0, "sat": 0.60, "accent_hue":  95.0},
}

CLASS_SHAPES: dict[str, str] = {
    "diabetes":           "icosahedron",
    "glaucoma":           "torus_knot",
    "multiple_sclerosis": "dodecahedron",
    "dry_eye":            "twisted_cylinder",
    "healthy":            "sphere",
}

SCALES: dict[str, tuple[int, ...]] = {
    "pentatonic_major": (0, 2, 4, 7, 9),
    "pentatonic_minor": (0, 3, 5, 7, 10),
    "natural_minor":    (0, 2, 3, 5, 7, 8, 10),
    "major":            (0, 2, 4, 5, 7, 9, 11),
    "lydian":           (0, 2, 4, 6, 7, 9, 11),
    "dorian":           (0, 2, 3, 5, 7, 9, 10),
    "phrygian":         (0, 1, 3, 5, 7, 8, 10),
    "hirajoshi":        (0, 2, 3, 7, 8),
    "kumoi":            (0, 2, 3, 7, 9),
}

INSTRUMENTS: tuple[str, ...] = (
    "piano",
    "guitar-nylon",
    "harp",
    "flute",
    "xylophone",
    "cello",
    "violin",
    "harmonium",
    "bassoon",
)

REVERBS: tuple[tuple[str, float, float], ...] = (
    ("small-room",   1.2, 0.22),
    ("wood-hall",    2.0, 0.30),
    ("stone-church", 3.4, 0.40),
    ("cathedral",    5.0, 0.46),
)


def _seed(item_id: str) -> int:
    return int.from_bytes(hashlib.sha1(item_id.encode()).digest()[:8], "big")


def rng_for(item_id: str) -> np.random.Generator:
    return np.random.default_rng(_seed(item_id))


def _hsl_to_rgb(h: float, s: float, l: float) -> tuple[int, int, int]:
    h = (h % 360.0) / 360.0
    if s <= 0:
        v = int(round(l * 255))
        return (v, v, v)
    a = s * min(l, 1 - l)

    def f(n: float) -> float:
        k = (n + h * 12) % 12
        return l - a * max(-1.0, min(k - 3, 9 - k, 1.0))

    return (
        int(round(f(0) * 255)),
        int(round(f(8) * 255)),
        int(round(f(4) * 255)),
    )


def class_lut(class_label: str, item_id: str) -> np.ndarray:
    """Return a (256, 3) uint8 LUT: shadow -> class-hue mid -> accent highlight.

    Hue anchor is fixed by class; per-item RNG shifts hue ±14°, saturation ±0.12,
    and lightness ±0.06, so cards within a class share a family without being
    identical. Accent hue comes from CLASS_PALETTES and is independently jittered.
    """
    anchor = CLASS_PALETTES.get(class_label) or CLASS_PALETTES["healthy"]
    rng = rng_for(item_id + ":lut")
    hue_shift = float(rng.uniform(-14.0, 14.0))
    sat_shift = float(rng.uniform(-0.12, 0.12))
    light_shift = float(rng.uniform(-0.06, 0.06))
    accent_shift = float(rng.uniform(-18.0, 18.0))

    base_h = float(anchor["hue"]) + hue_shift
    base_s = max(0.25, min(0.95, float(anchor["sat"]) + sat_shift))
    base_l = 0.55 + light_shift
    accent_h = float(anchor["accent_hue"]) + accent_shift

    shadow = _hsl_to_rgb(base_h, base_s * 0.7, 0.08)
    low    = _hsl_to_rgb(base_h, base_s, max(0.18, base_l - 0.32))
    mid    = _hsl_to_rgb(base_h, base_s, base_l)
    high   = _hsl_to_rgb(accent_h, base_s * 0.9, min(0.88, base_l + 0.28))
    top    = _hsl_to_rgb(accent_h, min(0.5, base_s * 0.6), 0.96)

    stops = np.array([shadow, low, mid, high, top], dtype=np.float32)
    positions = np.linspace(0.0, 1.0, len(stops))
    xs = np.linspace(0.0, 1.0, 256)
    lut = np.empty((256, 3), dtype=np.float32)
    for c in range(3):
        lut[:, c] = np.interp(xs, positions, stops[:, c])
    return np.clip(lut, 0, 255).astype(np.uint8)


def class_accent_css(class_label: str, item_id: str) -> str:
    """CSS hsl() string for this card's accent — used for borders/badges in the UI."""
    anchor = CLASS_PALETTES.get(class_label) or CLASS_PALETTES["healthy"]
    rng = rng_for(item_id + ":css")
    h = (float(anchor["hue"]) + float(rng.uniform(-14.0, 14.0))) % 360.0
    s = max(0.3, min(0.9, float(anchor["sat"]) + float(rng.uniform(-0.12, 0.12))))
    return f"hsl({h:.1f}, {s * 100:.0f}%, 62%)"


def _melodic_contour(features: list[float] | None, rng: np.random.Generator, length: int, scale_size: int) -> list[int]:
    """Melody degree indices in [0, scale_size). Features (row-mean quantiles of
    the height array) drive the coarse arc; RNG adds small ornament jitter."""
    if features and len(features) >= 3:
        arr = np.asarray(features, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        rng_range = max(1e-6, hi - lo)
        base = ((arr - lo) / rng_range * (scale_size - 1)).round().astype(int)
        if len(base) >= length:
            pick_idx = np.linspace(0, len(base) - 1, length).round().astype(int)
            base = base[pick_idx]
        else:
            reps = int(np.ceil(length / len(base)))
            base = np.tile(base, reps)[:length]
        jitter = rng.integers(-1, 2, size=length)
        out = np.clip(base + jitter, 0, scale_size - 1)
        return [int(v) for v in out]
    return [int(rng.integers(0, scale_size)) for _ in range(length)]


def audio_params_for(
    item_id: str,
    class_label: str,
    height_features: list[float] | None = None,
) -> dict[str, Any]:
    """Return a JSON-safe dict of deterministic per-item acoustic-audio parameters.

    ``height_features`` is an optional list of scalar descriptors (e.g. row-mean
    quantiles of the cleaned height array). When present it drives the melodic
    contour so the sound actually reflects the topology, not just the id.
    """
    rng = rng_for(item_id + ":audio")
    scale_name = str(rng.choice(list(SCALES.keys())))
    scale = SCALES[scale_name]
    instrument = str(rng.choice(INSTRUMENTS))

    root_midi = int(rng.integers(38, 68))
    tempo_bpm = float(rng.uniform(62.0, 148.0))
    length = int(rng.integers(7, 13))
    reverb_name, reverb_decay, reverb_wet = REVERBS[int(rng.integers(0, len(REVERBS)))]

    note_durations = list(rng.choice(
        a=[0.25, 0.5, 0.75, 1.0],
        size=length,
        p=[0.3, 0.4, 0.2, 0.1],
    ).astype(float))

    contour = _melodic_contour(height_features, rng, length, len(scale))
    octaves = [int(v) for v in rng.integers(-1, 2, size=length)]
    velocities = [float(v) for v in rng.uniform(0.5, 0.95, size=length)]
    pan_drift = float(rng.uniform(-0.35, 0.35))

    return {
        "instrument": instrument,
        "scale": scale_name,
        "scale_intervals": list(scale),
        "root_midi": root_midi,
        "tempo_bpm": tempo_bpm,
        "length": length,
        "reverb": reverb_name,
        "reverb_decay": reverb_decay,
        "reverb_wet": reverb_wet,
        "contour": contour,
        "octaves": octaves,
        "durations": note_durations,
        "velocities": velocities,
        "pan_drift": pan_drift,
    }


def shape_params_for(item_id: str, class_label: str) -> dict[str, Any]:
    """Procedural 3D shape: primitive fixed by class, deformation seeded by id."""
    rng = rng_for(item_id + ":shape")
    anchor = CLASS_PALETTES.get(class_label) or CLASS_PALETTES["healthy"]
    primitive = CLASS_SHAPES.get(class_label, "sphere")

    return {
        "primitive": primitive,
        "noise_seed": int(rng.integers(0, 2**31 - 1)),
        "noise_scale": float(rng.uniform(0.9, 2.6)),
        "noise_amp": float(rng.uniform(0.08, 0.28)),
        "displacement_amp": float(rng.uniform(0.15, 0.45)),
        "rotation_speed": float(rng.uniform(0.15, 0.45)),
        "rotation_tilt": float(rng.uniform(-0.35, 0.35)),
        "metalness": float(rng.uniform(0.05, 0.35)),
        "roughness": float(rng.uniform(0.25, 0.75)),
        "hue": float(anchor["hue"]) + float(rng.uniform(-14.0, 14.0)),
        "accent_hue": float(anchor["accent_hue"]) + float(rng.uniform(-18.0, 18.0)),
        "saturation": max(0.25, min(0.9, float(anchor["sat"]) + float(rng.uniform(-0.1, 0.1)))),
        "lightness": float(rng.uniform(0.48, 0.62)),
    }


def crop_plan_for(item_id: str, h: int, w: int) -> dict[str, int]:
    """Deterministic zoomed crop window: size and top-left corner from hash."""
    rng = rng_for(item_id + ":crop")
    choices = (128, 160, 192, 224, 256)
    # bias toward larger crops on small arrays to keep a usable view
    max_size = min(max(choices), min(h, w) - 4)
    valid = tuple(c for c in choices if c <= max_size) or (max_size,)
    size = int(rng.choice(valid))
    top = int(rng.integers(0, max(1, h - size)))
    left = int(rng.integers(0, max(1, w - size)))
    return {"top": top, "left": left, "size": size}


__all__ = [
    "CLASSES",
    "CLASS_PALETTES",
    "CLASS_SHAPES",
    "SCALES",
    "INSTRUMENTS",
    "REVERBS",
    "audio_params_for",
    "shape_params_for",
    "crop_plan_for",
    "class_lut",
    "class_accent_css",
    "rng_for",
]
