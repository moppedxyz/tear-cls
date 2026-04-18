"""Per-class accuracy + macro F1 for tear-cls. Called from train.py."""

from __future__ import annotations

from collections import Counter

from sklearn.metrics import f1_score

CLASSES = ["diabetes", "glaucoma", "multiple_sclerosis", "dry_eye", "healthy"]


def compute_metrics(gold: list[str], pred: list[str]) -> dict:
    if len(gold) != len(pred):
        raise ValueError(f"gold/pred length mismatch: {len(gold)} vs {len(pred)}")
    total: Counter[str] = Counter()
    correct: Counter[str] = Counter()
    for g, p in zip(gold, pred):
        total[g] += 1
        if g == p:
            correct[g] += 1
    per_class_acc = {
        cls: (correct[cls] / total[cls]) if total[cls] else float("nan")
        for cls in CLASSES
    }
    return {
        "accuracy": sum(g == p for g, p in zip(gold, pred)) / len(gold),
        "f1_macro": f1_score(gold, pred, labels=CLASSES, average="macro", zero_division=0),
        "per_class_acc": per_class_acc,
        "support": {cls: total[cls] for cls in CLASSES},
        "n": len(gold),
    }


def print_report(metrics: dict, split: str) -> None:
    print(f"\n=== {split} ({metrics['n']} samples) ===")
    print(f"accuracy : {metrics['accuracy']:.3f}")
    print(f"f1_macro : {metrics['f1_macro']:.3f}")
    print("per-class accuracy:")
    for cls in CLASSES:
        n = metrics["support"][cls]
        acc = metrics["per_class_acc"][cls]
        acc_str = f"{acc:.3f}" if n else "  n/a"
        print(f"  {cls:20s}  acc={acc_str}  n={n}")
