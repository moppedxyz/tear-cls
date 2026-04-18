"""Raw-Bruker feature extraction shared by `eda.ipynb` (training) and
`tearcls.server` (inference).

Every numeric feature the deployed classifier sees is produced here. The
notebook imports the same functions, so train-time and serve-time feature
vectors can't drift.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pySPM

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load-bearing column order. The trained sklearn pipeline was fit on a
# slice of these columns (see ckpt['selected_cols']) — reordering here
# silently invalidates every saved model.
FEATURE_COLS: tuple[str, ...] = (
    "scan_size_um",
    "samps_line",
    "nm_per_px",
    "z_range_nm",
    "z_std_nm",
    "Ra_nm",
    "Rq_nm",
    "Rsk",
    "Rku",
    "PSD_slope",
    "Sdr",
)

TARGET_LABELS: list[str] = ["healthy", "dry_eye", "other_disease"]

TARGET_MAP: dict[str, str] = {
    "healthy":            "healthy",
    "dry_eye":            "dry_eye",
    "diabetes":           "other_disease",
    "glaucoma":           "other_disease",
    "multiple_sclerosis": "other_disease",
}

# Same regex as `tearcls/data_split.py:30` — keep in sync. Strips the
# rendered-BMP suffix so we land on the extension-less raw twin.
_BMP_SUFFIX_RE = re.compile(r"_1\.bmp$|_\.bmp$|\.bmp$")


def _dec(d: dict, k: str) -> str | None:
    v = d.get(k.encode())
    if not v:
        return None
    return v[0].decode("latin-1", errors="ignore").strip()


def extract_header(path: str | Path) -> dict:
    """Parse a Bruker raw file via pySPM. Returns scan geometry + Z stats.

    Keys: scan_size_um, samps_line, lines, scanner_type, head_type, serial,
    z_range_nm, z_std_nm, nm_per_px. Per-step failures yield NaN/None for
    that key; the function never raises on a valid NanoScope file.
    """
    scan = pySPM.Bruker(str(path))
    l0 = scan.layers[0]

    scan_size_um = None
    for layer in scan.layers:
        s = _dec(layer, "Scan Size")
        if s:
            try:
                scan_size_um = float(s.split()[0])
            except Exception:
                pass
            break

    out: dict = {
        "scan_size_um": scan_size_um,
        "samps_line":   int(_dec(l0, "Samps/line")      or 0) or None,
        "lines":        int(_dec(l0, "Number of lines") or 0) or None,
        "scanner_type": None,
        "head_type":    None,
        "serial":       None,
        "z_range_nm":   np.nan,
        "z_std_nm":     np.nan,
    }
    if scan.scanners:
        sc = scan.scanners[0]
        out["scanner_type"] = _dec(sc, "Scanner type")
        out["head_type"]    = _dec(sc, "Head Type")
        out["serial"]       = _dec(sc, "Serial Number")
    try:
        ch = scan.get_channel("Height Sensor").correct_plane()
        px = ch.pixels
        out["z_range_nm"] = float(px.max() - px.min())
        out["z_std_nm"]   = float(px.std())
    except Exception:
        pass
    if out.get("samps_line") and out.get("scan_size_um"):
        out["nm_per_px"] = 1000.0 * out["scan_size_um"] / out["samps_line"]
    else:
        out["nm_per_px"] = np.nan
    return out


def _flatten_channel(scan: pySPM.Bruker, name: str) -> np.ndarray | None:
    try:
        ch = scan.get_channel(name)
    except Exception:
        return None
    try:
        ch = ch.correct_lines()
    except Exception:
        pass
    try:
        ch = ch.correct_plane()
    except Exception:
        pass
    return ch.pixels.astype(np.float32)


def _psd_slope(z: np.ndarray) -> float:
    F = np.fft.fftshift(np.fft.fft2(z - z.mean()))
    P = np.abs(F) ** 2
    h, w = P.shape
    y, x = np.indices(P.shape)
    r = np.hypot(y - h / 2, x - w / 2).astype(int)
    rmax = min(h, w) // 2
    rad = np.bincount(r.ravel(), weights=P.ravel())[:rmax]
    cnt = np.bincount(r.ravel())[:rmax]
    psd = rad / np.maximum(cnt, 1)
    k = np.arange(2, rmax)
    yv = np.log(psd[k] + 1e-12)
    xv = np.log(k.astype(float))
    slope, _ = np.polyfit(xv, yv, 1)
    return float(slope)


def roughness(z: np.ndarray, nm_per_px: float | None) -> dict:
    """Compute {Ra_nm, Rq_nm, Rsk, Rku, PSD_slope, Sdr} from a 2D height map.

    Sdr requires a positive finite nm_per_px; otherwise returns NaN for Sdr.
    """
    mean = z.mean()
    res  = z - mean
    ra  = float(np.abs(res).mean())
    rq  = float(np.sqrt((res ** 2).mean()))
    rsk = float(((res / (rq + 1e-9)) ** 3).mean())
    rku = float(((res / (rq + 1e-9)) ** 4).mean())
    if nm_per_px and np.isfinite(nm_per_px) and nm_per_px > 0:
        dzx = np.diff(z, axis=1) / nm_per_px
        dzy = np.diff(z, axis=0) / nm_per_px
        h = min(dzx.shape[0], dzy.shape[0])
        w = min(dzx.shape[1], dzy.shape[1])
        area_ratio = float(
            np.sqrt(1 + dzx[:h, :w] ** 2 + dzy[:h, :w] ** 2).mean()
        )
        sdr = area_ratio - 1.0
    else:
        sdr = float("nan")
    return {
        "Ra_nm": ra, "Rq_nm": rq, "Rsk": rsk, "Rku": rku,
        "PSD_slope": _psd_slope(z), "Sdr": sdr,
    }


def extract_features(raw_path: str | Path) -> dict[str, float]:
    """Return a dict keyed by FEATURE_COLS for one raw Bruker scan.

    Single pySPM open. Tries the `Height Sensor` channel first (sensor read,
    closer to truth) then falls back to `Height` (closed-loop estimate).
    Missing values come back as NaN; the model loader fills them via the
    per-column training medians stored in the checkpoint.
    """
    hdr = extract_header(raw_path)
    scan = pySPM.Bruker(str(raw_path))
    z = _flatten_channel(scan, "Height Sensor")
    if z is None:
        z = _flatten_channel(scan, "Height")
    rough = roughness(z, hdr.get("nm_per_px")) if z is not None else {
        c: float("nan") for c in ("Ra_nm", "Rq_nm", "Rsk", "Rku", "PSD_slope", "Sdr")
    }
    merged = {**hdr, **rough}
    return {c: float(merged.get(c)) if merged.get(c) is not None else float("nan")
            for c in FEATURE_COLS}


def vectorize(
    feat_dict: dict[str, float],
    fallback: dict[str, float] | None = None,
) -> np.ndarray:
    """Order a feature dict into the FEATURE_COLS vector.

    NaN policy: substitute `fallback[col]` if provided (the per-column
    training medians persisted with the checkpoint), else 0.0 with a warning
    listing which columns were missing — at inference we have N=1, so no
    sample-level median exists.
    """
    fallback = fallback or {}
    missing: list[str] = []
    out = np.empty(len(FEATURE_COLS), dtype=np.float32)
    for i, col in enumerate(FEATURE_COLS):
        v = feat_dict.get(col, float("nan"))
        if v is None or not np.isfinite(v):
            fb = fallback.get(col)
            if fb is None or not np.isfinite(fb):
                missing.append(col)
                v = 0.0
            else:
                v = float(fb)
        out[i] = v
    if missing:
        log.warning("vectorize: NaN with no fallback for %s; defaulting to 0", missing)
    return out


def derive_raw_path(bmp_relpath: str, repo_root: Path | None = None) -> Path:
    """`data_raw/Diabetes/37_DM.010_1.bmp` -> absolute path to
    `data_raw/Diabetes/37_DM.010`.

    The `.NNN` sequential suffix stays — only the BMP-render suffix is
    stripped. Mirrors the suffix regex at `tearcls/data_split.py:30`.
    """
    root = repo_root or REPO_ROOT
    stripped = _BMP_SUFFIX_RE.sub("", bmp_relpath)
    p = Path(stripped)
    return p if p.is_absolute() else (root / p)
