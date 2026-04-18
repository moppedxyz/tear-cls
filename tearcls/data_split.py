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
    "Diabetes":          "diabetes",
    "PGOV_Glaukom":      "glaucoma",
    "SklerózaMultiplex": "multiple_sclerosis",
    "SucheOko":          "dry_eye",
    "ZdraviLudia":       "healthy",
}

SEED = 3407
VAL_FRAC = 0.1
TEST_FRAC = 0.1


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
            rows.append({
                "filepath":     bmp.relative_to(REPO_ROOT).as_posix(),
                "class":        class_folder,
                "label":        CLASS_MAP[class_folder],
                "patient_code": parse_patient_code(bmp.name),
            })
    rows.sort(key=lambda r: r["filepath"])
    return rows


def assign_splits(rows: list[dict]) -> None:
    """Split per class: for each class, allocate that class's unique patients
    to train/val/test at ~80/10/10. Guarantees every class appears in every
    split AND no patient crosses splits.

    Fallback: a class with fewer than 3 unique patients can't be 3-way-split
    by group, so it falls back to an image-level split for that class (logs a
    WARN so the leakage is visible in the summary)."""
    rng = random.Random(SEED)

    rows_by_class: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        rows_by_class[r["class"]].append(i)

    for cls in CLASS_MAP:
        class_indices = rows_by_class[cls]
        patient_to_rows: dict[str, list[int]] = defaultdict(list)
        for i in class_indices:
            patient_to_rows[rows[i]["patient_code"]].append(i)

        patients = sorted(patient_to_rows.keys())
        rng.shuffle(patients)

        if len(patients) < 3:
            print(f"WARN: class {cls} has only {len(patients)} patient(s); "
                  f"falling back to image-level split within the class")
            imgs = sorted(class_indices)
            rng.shuffle(imgs)
            m = len(imgs)
            n_test = max(1, round(m * TEST_FRAC))
            n_val = max(1, round(m * VAL_FRAC))
            n_train = m - n_test - n_val
            for i in imgs[:n_train]:
                rows[i]["split"] = "train"
            for i in imgs[n_train:n_train + n_val]:
                rows[i]["split"] = "val"
            for i in imgs[n_train + n_val:]:
                rows[i]["split"] = "test"
            continue

        n = len(patients)
        n_test = max(1, round(n * TEST_FRAC))
        n_val = max(1, round(n * VAL_FRAC))
        test_groups = patients[:n_test]
        val_groups = patients[n_test:n_test + n_val]
        train_groups = patients[n_test + n_val:]

        for p in train_groups:
            for i in patient_to_rows[p]:
                rows[i]["split"] = "train"
        for p in val_groups:
            for i in patient_to_rows[p]:
                rows[i]["split"] = "val"
        for p in test_groups:
            for i in patient_to_rows[p]:
                rows[i]["split"] = "test"


def summarize(rows: list[dict]) -> None:
    total = len(rows)
    by_split = Counter(r["split"] for r in rows)
    per_class_split: dict[str, Counter] = defaultdict(Counter)
    patients_by_split: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        per_class_split[r["class"]][r["split"]] += 1
        patients_by_split[r["split"]].add(r["patient_code"])

    print(f"Total: {total} images across {len(CLASS_MAP)} classes")
    print(f"Split: train={by_split['train']}  val={by_split['val']}  test={by_split['test']}")
    print("Per-class (train/val/test):")
    for cls in CLASS_MAP:
        c = per_class_split[cls]
        print(f"  {cls:20s}  train={c['train']:3d}  val={c['val']:3d}  test={c['test']:3d}")

    tr, va, te = patients_by_split["train"], patients_by_split["val"], patients_by_split["test"]
    total_patients = len(tr | va | te)
    print(
        f"Patient leakage  train∩val={len(tr & va)}  train∩test={len(tr & te)}  "
        f"val∩test={len(va & te)}  (total unique patients: {total_patients})"
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
    assign_splits(rows)
    write_csv(rows)
    summarize(rows)


if __name__ == "__main__":
    main()
