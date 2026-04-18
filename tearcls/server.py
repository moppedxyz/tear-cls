"""UI server: artistic AFM gallery, per-card sonification, 5-class classifier.

Gallery items come from ``data/processed/index.csv`` (variant == "original" —
one card per real sample, not per augmentation). Cleaned height arrays come
from ``data/gwy/<ClassFolder>/<stem>.npy`` (produced by ``tearcls.gwy_prep``);
if a .npy is missing the server falls back to pySPM + in-process levelling.

No ``.bmp`` is served to the browser — cards are rendered by
``tearcls.render.compose_card`` (deterministic zoomed crop + lambert shading +
class-tinted LUT from ``tearcls.palettes``).

Classification uses the sklearn pipeline in ``checkpoints/best.joblib``
(trained on raw-Bruker features via ``tearcls.raw_features``). Feature
extraction runs on the paired raw-Bruker binary — also not a ``.bmp``.

Routes:
    GET  /                           -> index.html
    GET  /ui/*                       -> static UI assets (app.js, styles.css)
    GET  /gallery                    -> JSON list of card items
    GET  /gallery/{id}/image         -> full-res artistic PNG (cached to disk)
    GET  /gallery/{id}/thumb         -> small card thumbnail PNG
    GET  /gallery/{id}/heightmap     -> grayscale displacement texture
    GET  /gallery/{id}/audio-params  -> JSON (deterministic acoustic params)
    GET  /gallery/{id}/shape-params  -> JSON (procedural 3D shape params)
    POST /gallery/{id}/classify      -> {prediction, probabilities}
"""

from __future__ import annotations

import csv
import hashlib
import io
import logging
import re
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from PIL import Image

from tearcls import render as render_mod
from tearcls.data_split import CLASS_MAP
from tearcls.gwy_prep import level_pure_python, _read_bruker_height
from tearcls.palettes import (
    audio_params_for,
    class_accent_css,
    shape_params_for,
)
from tearcls.raw_features import (
    FEATURE_COLS,
    TARGET_LABELS,
    extract_features,
    vectorize,
)

log = logging.getLogger("tearcls.server")

REPO_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = Path(__file__).parent / "ui"
INDEX_HTML = UI_DIR / "index.html"
CKPT_PATH = REPO_ROOT / "checkpoints" / "best.joblib"

PROCESSED_INDEX = REPO_ROOT / "data" / "processed" / "index.csv"
GWY_ROOT = REPO_ROOT / "data" / "gwy"
CACHE_ROOT = REPO_ROOT / "data" / "cache" / "renders"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)

_BMP_SUFFIX_RE = re.compile(r"_1\.bmp$|_\.bmp$|\.bmp$")

app = FastAPI(title="Tear Gallery")
app.mount("/ui", StaticFiles(directory=str(UI_DIR)), name="ui")

_MODEL: dict = {"loaded": False}
_ITEMS: list[dict] = []
_ITEMS_BY_ID: dict[str, dict] = {}
_HEIGHT_CACHE: dict[str, np.ndarray] = {}
_CLASSIFY_CACHE: dict[str, dict] = {}


def _load_model() -> None:
    if not CKPT_PATH.exists():
        log.warning("no checkpoint at %s — predictions disabled", CKPT_PATH)
        return
    try:
        ckpt = joblib.load(CKPT_PATH)
    except Exception as e:  # noqa: BLE001
        log.exception("failed to load %s: %s", CKPT_PATH, e)
        return
    _MODEL.update(
        loaded=True,
        pipe=ckpt["pipe"],
        classes=ckpt.get("classes", TARGET_LABELS),
        selected_cols=ckpt["selected_cols"],
        feature_medians=ckpt.get("feature_medians", {}),
        cv_macro_f1=ckpt.get("cv_macro_f1"),
        trained_at=ckpt.get("trained_at"),
    )
    log.info(
        "loaded %s (selected=%d/%d, cv_f1=%s, trained=%s)",
        CKPT_PATH, len(_MODEL["selected_cols"]), len(FEATURE_COLS),
        _MODEL["cv_macro_f1"], _MODEL["trained_at"],
    )


def _derive_raw_bruker(source_filepath: str) -> Path:
    """`data_raw/Diabetes/37_DM.010_1.bmp` -> `data_raw/Diabetes/37_DM.010`."""
    return REPO_ROOT / _BMP_SUFFIX_RE.sub("", source_filepath)


def _scan_gallery() -> None:
    _ITEMS.clear()
    _ITEMS_BY_ID.clear()
    if not PROCESSED_INDEX.exists():
        log.warning("no %s — run tearcls/augmentation.py first; gallery empty", PROCESSED_INDEX)
        return

    cls_folder_by_label = {v: k for k, v in CLASS_MAP.items()}

    with PROCESSED_INDEX.open() as f:
        rows = list(csv.DictReader(f))

    for row in rows:
        if row.get("variant") != "original":
            continue
        class_folder = row["class"]
        label5 = row["label"]
        src_bmp = row["source_filepath"]
        stem = _BMP_SUFFIX_RE.sub("", Path(src_bmp).name)
        item_id = hashlib.sha1(f"{class_folder}/{stem}".encode()).hexdigest()[:12]
        item = {
            "id": item_id,
            "class_folder": class_folder,
            "label": label5,
            "patient_code": row.get("patient_code", ""),
            "split": row.get("split", ""),
            "stem": stem,
            "processed_path": row["filepath"],
            "source_bmp": src_bmp,
            "raw_path": _derive_raw_bruker(src_bmp).as_posix(),
            "npy_path": (GWY_ROOT / class_folder / f"{stem}.npy").as_posix(),
        }
        _ITEMS.append(item)
        _ITEMS_BY_ID[item_id] = item

    log.info("gallery: %d items from %s", len(_ITEMS), PROCESSED_INDEX)


def _load_height(item: dict) -> np.ndarray:
    """Return a 2-D float32 height array for an item. Prefers the pre-cleaned
    ``.npy`` from ``gwy_prep``; falls back to in-process pySPM + levelling;
    last-resort falls back to the processed-PNG luminance (which is already
    colormapped, so this produces a degraded but non-broken display)."""
    cached = _HEIGHT_CACHE.get(item["id"])
    if cached is not None:
        return cached

    npy = Path(item["npy_path"])
    if npy.exists():
        z = np.load(npy).astype(np.float32)
        _HEIGHT_CACHE[item["id"]] = z
        return z

    raw = Path(item["raw_path"])
    if raw.exists():
        raw_z = _read_bruker_height(raw)
        if raw_z is not None:
            z = level_pure_python(raw_z)
            _HEIGHT_CACHE[item["id"]] = z
            return z

    png = REPO_ROOT / item["processed_path"]
    if png.exists():
        img = Image.open(png).convert("L")
        z = np.asarray(img, dtype=np.float32)
        log.warning("fell back to processed-PNG luminance for item %s", item["id"])
        _HEIGHT_CACHE[item["id"]] = z
        return z

    raise HTTPException(500, f"no height source for item {item['id']!r}")


def _png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def _require_item(item_id: str) -> dict:
    item = _ITEMS_BY_ID.get(item_id)
    if item is None:
        raise HTTPException(404, f"gallery item {item_id!r} not found")
    return item


@app.on_event("startup")
def _boot() -> None:
    _load_model()
    _scan_gallery()


@app.get("/")
def index() -> FileResponse:
    if not INDEX_HTML.exists():
        raise HTTPException(500, f"missing {INDEX_HTML}")
    return FileResponse(INDEX_HTML, media_type="text/html")


@app.get("/gallery")
def gallery_list() -> JSONResponse:
    items_out = []
    for it in _ITEMS:
        items_out.append({
            "id": it["id"],
            "class_folder": it["class_folder"],
            "label": it["label"],
            "patient_code": it["patient_code"],
            "split": it["split"],
            "stem": it["stem"],
            "accent_css": class_accent_css(it["label"], it["id"]),
        })
    return JSONResponse({
        "items": items_out,
        "count": len(items_out),
        "classes": list(CLASS_MAP.values()),
        "classifier_classes": list(_MODEL.get("classes") or TARGET_LABELS),
    })


@app.get("/gallery/{item_id}/image")
def gallery_image(item_id: str, size: int = 1024) -> Response:
    item = _require_item(item_id)
    size = max(256, min(2048, int(size)))
    cache_path = CACHE_ROOT / f"{item_id}_{size}.png"
    if cache_path.exists():
        return FileResponse(cache_path, media_type="image/png")
    z = _load_height(item)
    img = render_mod.compose_card(z, item_id, item["label"], target_size=size)
    cache_path.write_bytes(_png_bytes(img))
    return FileResponse(cache_path, media_type="image/png")


@app.get("/gallery/{item_id}/thumb")
def gallery_thumb(item_id: str) -> Response:
    return gallery_image(item_id, size=384)


@app.get("/gallery/{item_id}/heightmap")
def gallery_heightmap(item_id: str) -> Response:
    item = _require_item(item_id)
    z = _load_height(item)
    img = render_mod.render_heightmap_png(z, size=256)
    return Response(
        content=_png_bytes(img),
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=3600"},
    )


@app.get("/gallery/{item_id}/audio-params")
def gallery_audio(item_id: str) -> JSONResponse:
    item = _require_item(item_id)
    z = _load_height(item)
    crop = render_mod.sample_crop(z, item_id)
    feats = render_mod.height_features(crop)
    params = audio_params_for(item_id, item["label"], feats)
    return JSONResponse(params)


@app.get("/gallery/{item_id}/shape-params")
def gallery_shape(item_id: str) -> JSONResponse:
    item = _require_item(item_id)
    return JSONResponse(shape_params_for(item_id, item["label"]))


@app.post("/gallery/{item_id}/classify")
def gallery_classify(item_id: str) -> JSONResponse:
    item = _require_item(item_id)
    cached = _CLASSIFY_CACHE.get(item_id)
    if cached is not None:
        return JSONResponse(cached)

    if not _MODEL["loaded"]:
        return JSONResponse({
            "prediction": None,
            "probabilities": {},
            "note": "classifier checkpoint not loaded — predictions disabled",
        })

    raw = Path(item["raw_path"])
    if not raw.exists():
        return JSONResponse({
            "prediction": None,
            "probabilities": {},
            "note": f"raw Bruker file missing: {item['raw_path']}",
        })

    try:
        features = extract_features(raw)
    except Exception as e:  # noqa: BLE001
        log.exception("feature extraction failed for %s", raw)
        raise HTTPException(500, f"feature extraction failed: {e}") from e

    full = vectorize(features, fallback=_MODEL["feature_medians"])
    sel_idx = [FEATURE_COLS.index(c) for c in _MODEL["selected_cols"]]
    x = full[sel_idx][None, :]
    probs = _MODEL["pipe"].predict_proba(x)[0]
    classes = list(_MODEL["classes"])
    top = classes[int(np.argmax(probs))]
    out = {
        "prediction": top,
        "probabilities": {c: float(p) for c, p in zip(classes, probs)},
        "classifier_classes": classes,
        "ground_truth_label": item["label"],
    }
    _CLASSIFY_CACHE[item_id] = out
    return JSONResponse(out)


def main() -> None:
    import uvicorn
    uvicorn.run("tearcls.server:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
