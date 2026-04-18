import csv
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = REPO_ROOT / "data_raw"
OUT_DIR = REPO_ROOT / "data"
OUT_CSV = OUT_DIR / "splits.csv"

CLASS_MAP: dict[str, str] = {
    "Diabetes": "diabetes",
    "PGOV_Glaukom": "glaucoma",
    "SklerózaMultiplex": "multiple_sclerosis",
    "SucheOko": "dry_eye",
    "ZdraviLudia": "healthy",
}

SEED = 3407
EVAL_FRAC = 0.2  # held-out patients per class; val and test share this set


def parse_patient_code(bmp_name: str) -> str:
    """`37_DM.010_1.bmp` → `37_DM`. Cut on the first `.NNN` numeric segment so
    patient IDs with embedded dates (e.g. `DM_01.03.2024_LO.001_1`) still
    extract cleanly as `DM_01.03.2024_LO`."""
    stem = re.sub(r"_1\.bmp$|_\.bmp$|\.bmp$", "", bmp_name)
    cut = re.split(r"\.\d+", stem, maxsplit=1)[0]
    return cut or stem


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for class_folder in CLASS_MAP:
        folder = RAW_DIR / class_folder
        if not folder.is_dir():
            sys.exit(f"Missing class folder: {folder}")
        for bmp in sorted(folder.glob("*.bmp")):
            rows.append(
                {
                    "filepath": bmp.relative_to(REPO_ROOT).as_posix(),
                    "class": class_folder,
                    "label": CLASS_MAP[class_folder],
                    "patient_code": parse_patient_code(bmp.name),
                }
            )
    rows.sort(key=lambda r: r["filepath"])
    return rows


def assign_splits(rows: list[dict]) -> list[dict]:
    """Split per class: for each class, allocate that class's unique patients
    to train vs. a held-out set at ~80/20. Guarantees every class appears in
    train AND held-out (when it has ≥2 patients) AND no patient crosses
    splits.

    Val and test are merged into a single held-out set — with only a few
    unique patients per class there aren't enough to support three disjoint
    held-out groups without leakage. Each held-out row is emitted twice in
    the output: once with split="val", once with split="test" — so downstream
    loaders that expect both splits still work, and the two sets are
    identical by construction.

    Returns the expanded row list (not in-place)."""
    rng = random.Random(SEED)

    rows_by_class: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        rows_by_class[r["class"]].append(i)

    out: list[dict] = []
    for cls in CLASS_MAP:
        class_indices = rows_by_class[cls]
        patient_to_rows: dict[str, list[int]] = defaultdict(list)
        for i in class_indices:
            patient_to_rows[rows[i]["patient_code"]].append(i)

        patients = sorted(patient_to_rows.keys())
        rng.shuffle(patients)

        if len(patients) == 1:
            for i in patient_to_rows[patients[0]]:
                out.append({**rows[i], "split": "train"})
            print(
                f"WARN: class {cls} has 1 patient; entire class routed to "
                f"train (no val/test signal for this class)"
            )
            continue

        n = len(patients)
        n_eval = max(1, round(n * EVAL_FRAC))
        eval_patients = patients[:n_eval]
        train_patients = patients[n_eval:]

        for p in train_patients:
            for i in patient_to_rows[p]:
                out.append({**rows[i], "split": "train"})
        for p in eval_patients:
            for i in patient_to_rows[p]:
                out.append({**rows[i], "split": "val"})
                out.append({**rows[i], "split": "test"})

    return out


def summarize(rows: list[dict]) -> None:
    total = len(rows)
    by_split = Counter(r["split"] for r in rows)
    per_class_split: dict[str, Counter] = defaultdict(Counter)
    patients_by_split: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        per_class_split[r["class"]][r["split"]] += 1
        patients_by_split[r["split"]].add(r["patient_code"])

    print(f"Total: {total} images across {len(CLASS_MAP)} classes")
    print(
        f"Split: train={by_split['train']}  val={by_split['val']}  test={by_split['test']}"
    )
    print("Per-class (train/val/test):")
    for cls in CLASS_MAP:
        c = per_class_split[cls]
        print(
            f"  {cls:20s}  train={c['train']:3d}  val={c['val']:3d}  test={c['test']:3d}"
        )

    tr, va, te = (
        patients_by_split["train"],
        patients_by_split["val"],
        patients_by_split["test"],
    )
    total_patients = len(tr | va | te)
    print(
        f"Patient leakage  train∩val={len(tr & va)}  train∩test={len(tr & te)}  "
        f"(total unique patients: {total_patients}; val==test by design)"
    )


def write_csv(rows: list[dict]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = ["filepath", "class", "label", "patient_code", "split"]
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {OUT_CSV}  ({len(rows)} rows)")


def main() -> None:
    rows = collect_rows()
    rows = assign_splits(rows)
    write_csv(rows)
    summarize(rows)


if __name__ == "__main__":
    main()
