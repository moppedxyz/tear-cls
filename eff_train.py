"""Fine-tune a torchvision EfficientNet on AFM tear-film classification.

Reads the patient-stratified splits + pre-dumped augmented PNGs via
``tearcls.data.build_loaders`` (no runtime augmentation), trains with
ImageNet-pretrained weights, logs to W&B (``ramang/tear-cls`` by default),
selects the best checkpoint by val macro-F1, and reports test metrics.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
import torch.nn as nn
import torchvision
from torchvision.ops import StochasticDepth
import wandb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torchvision.transforms import v2 as T
from tqdm import tqdm

from tearcls.data import CLASSES, build_loaders

REPO_ROOT = Path(__file__).resolve().parent
SEED = 3407

MODEL_REGISTRY: dict[str, tuple] = {
    "efficientnet_b0": (
        torchvision.models.efficientnet_b0,
        torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1,
        224,
    ),
    "efficientnet_b1": (
        torchvision.models.efficientnet_b1,
        torchvision.models.EfficientNet_B1_Weights.IMAGENET1K_V1,
        240,
    ),
    "efficientnet_b3": (
        torchvision.models.efficientnet_b3,
        torchvision.models.EfficientNet_B3_Weights.IMAGENET1K_V1,
        300,
    ),
    "efficientnet_v2_s": (
        torchvision.models.efficientnet_v2_s,
        torchvision.models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,
        384,
    ),
    "efficientnet_v2_m": (
        torchvision.models.efficientnet_v2_m,
        torchvision.models.EfficientNet_V2_M_Weights.IMAGENET1K_V1,
        480,
    ),
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_model(
    model_name: str,
    num_classes: int = 5,
    head_dropout: float = 0.4,
    drop_path_mult: float = 1.5,
    drop_path_cap: float = 0.3,
) -> tuple[nn.Module, int]:
    if model_name not in MODEL_REGISTRY:
        raise SystemExit(f"Unknown model {model_name!r}. Choose from: {list(MODEL_REGISTRY)}")
    ctor, weights, img_size = MODEL_REGISTRY[model_name]
    model = ctor(weights=weights)
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    for m in model.classifier.modules():
        if isinstance(m, nn.Dropout):
            m.p = head_dropout
    for m in model.modules():
        if isinstance(m, StochasticDepth):
            m.p = min(drop_path_cap, m.p * drop_path_mult)
    return model, img_size


def make_collate(img_size: int):
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.ToImage(),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=list(IMAGENET_MEAN), std=list(IMAGENET_STD)),
    ])

    def collate(batch: list[dict]) -> tuple[torch.Tensor, torch.Tensor]:
        xs = torch.stack([transform(item["image"]) for item in batch])
        ys = torch.tensor([item["label_idx"] for item in batch], dtype=torch.long)
        return xs, ys

    return collate


def cosine_warmup_lambda(total_steps: int, warmup_steps: int):
    def fn(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return fn


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds: list[int] = []
    all_gold: list[int] = []
    all_probs: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        n += x.size(0)
        probs = torch.softmax(logits.float(), dim=1)
        all_preds.extend(logits.argmax(dim=1).cpu().tolist())
        all_gold.extend(y.cpu().tolist())
        all_probs.append(probs.cpu().numpy())
    probs_arr = np.concatenate(all_probs, axis=0) if all_probs else np.zeros((0, len(CLASSES)))
    return total_loss / max(1, n), np.array(all_gold), np.array(all_preds), probs_arr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="efficientnet_v2_s", choices=list(MODEL_REGISTRY))
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--head-lr-mult", type=float, default=10.0,
                    help="Head LR = lr * head_lr_mult. Backbone uses lr.")
    ap.add_argument("--freeze-epochs", type=int, default=2,
                    help="Freeze backbone (train head only) for the first N epochs.")
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--head-dropout", type=float, default=0.4)
    ap.add_argument("--drop-path-mult", type=float, default=1.5)
    ap.add_argument("--drop-path-cap", type=float, default=0.3)
    ap.add_argument("--mixup-alpha", type=float, default=0.2)
    ap.add_argument("--cutmix-alpha", type=float, default=1.0)
    ap.add_argument("--no-mixcut", action="store_true",
                    help="Disable MixUp/CutMix augmentation.")
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--class-weights", default="balanced",
                    choices=["none", "balanced", "effective"],
                    help="CE class weights. 'balanced' = inverse freq normalized to mean 1. "
                         "'effective' = Cui et al. effective-number weighting (beta=0.999).")
    ap.add_argument("--class-weights-beta", type=float, default=0.999,
                    help="Beta for 'effective' class weighting.")
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--eval-every-epochs", type=int, default=5,
                    help="Run val eval every N epochs (and always on the final epoch). "
                         "Default 5.")
    ap.add_argument("--no-augmented", action="store_true",
                    help="Train on originals only (val/test are originals either way).")
    ap.add_argument("--wandb-entity", default="ramang")
    ap.add_argument("--wandb-project", default="tear-cls")
    ap.add_argument("--wandb-run-name", default=None)
    ap.add_argument("--wandb-mode", default="online", choices=["online", "offline", "disabled"])
    args = ap.parse_args()

    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    out_dir = REPO_ROOT / "outputs" / args.model
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.environ["WANDB_MODE"] = args.wandb_mode
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        name=args.wandb_run_name or f"{args.model}-tearcls-data",
        mode=args.wandb_mode,
        job_type="train",
        config={
            "model": args.model,
            "method": "torchvision full fine-tune",
            "data_path": "tearcls.data.build_loaders",
            "load_augmented": not args.no_augmented,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "head_lr_mult": args.head_lr_mult,
            "freeze_epochs": args.freeze_epochs,
            "weight_decay": args.weight_decay,
            "head_dropout": args.head_dropout,
            "drop_path_mult": args.drop_path_mult,
            "drop_path_cap": args.drop_path_cap,
            "mixup_alpha": args.mixup_alpha,
            "cutmix_alpha": args.cutmix_alpha,
            "mixcut_enabled": not args.no_mixcut,
            "label_smoothing": args.label_smoothing,
            "class_weights": args.class_weights,
            "class_weights_beta": args.class_weights_beta,
            "warmup_ratio": args.warmup_ratio,
            "eval_every_epochs": args.eval_every_epochs,
            "seed": SEED,
            "classes": CLASSES,
        },
    )

    model, img_size = build_model(
        args.model,
        num_classes=len(CLASSES),
        head_dropout=args.head_dropout,
        drop_path_mult=args.drop_path_mult,
        drop_path_cap=args.drop_path_cap,
    )
    model.to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Model: {args.model} | input {img_size}x{img_size} | params {trainable/1e6:.1f}M / {total/1e6:.1f}M")
    wandb.log({
        "model/params_trainable": trainable,
        "model/params_total": total,
        "model/img_size": img_size,
    })

    train_dl, val_dl, test_dl = build_loaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=make_collate(img_size),
        load_augmented=not args.no_augmented,
    )

    def count_labels(dl) -> Counter:
        return Counter(r["label"] for r in dl.dataset.rows)

    train_counts = count_labels(train_dl)
    val_counts = count_labels(val_dl)
    test_counts = count_labels(test_dl)
    print(f"Train: {len(train_dl.dataset)} samples — {dict(train_counts)}")
    print(f"Val:   {len(val_dl.dataset)} samples — {dict(val_counts)}")
    print(f"Test:  {len(test_dl.dataset)} samples — {dict(test_counts)}")
    wandb.log({
        "data/train_size": len(train_dl.dataset),
        "data/val_size": len(val_dl.dataset),
        "data/test_size": len(test_dl.dataset),
        **{f"data/train_count/{k}": v for k, v in train_counts.items()},
        **{f"data/val_count/{k}": v for k, v in val_counts.items()},
        **{f"data/test_count/{k}": v for k, v in test_counts.items()},
    })

    counts = np.array([train_counts.get(c, 0) for c in CLASSES], dtype=np.float64)
    if args.class_weights == "none" or counts.sum() == 0:
        class_weight_t = None
    elif args.class_weights == "balanced":
        inv = 1.0 / np.maximum(counts, 1)
        w = inv * (len(CLASSES) / inv.sum())
        class_weight_t = torch.tensor(w, dtype=torch.float32, device=device)
    else:
        beta = args.class_weights_beta
        eff_num = 1.0 - np.power(beta, np.maximum(counts, 1))
        w = (1.0 - beta) / eff_num
        w = w * (len(CLASSES) / w.sum())
        class_weight_t = torch.tensor(w, dtype=torch.float32, device=device)
    if class_weight_t is not None:
        weight_map = {c: float(w) for c, w in zip(CLASSES, class_weight_t.cpu().tolist())}
        print(f"Class weights ({args.class_weights}): {weight_map}")
        wandb.log({f"data/class_weight/{k}": v for k, v in weight_map.items()})

    criterion = nn.CrossEntropyLoss(
        weight=class_weight_t, label_smoothing=args.label_smoothing
    )

    head_params = list(model.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": args.lr},
            {"params": head_params,     "lr": args.lr * args.head_lr_mult},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_dl) * args.epochs)
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, cosine_warmup_lambda(total_steps, warmup_steps)
    )

    use_mixcut = not args.no_mixcut
    if use_mixcut:
        mixcut = T.RandomChoice([
            T.MixUp(alpha=args.mixup_alpha, num_classes=len(CLASSES)),
            T.CutMix(alpha=args.cutmix_alpha, num_classes=len(CLASSES)),
        ])

    best_f1 = -1.0
    best_epoch = -1
    best_path = out_dir / "best.pt"
    global_step = 0
    best_val_gold: np.ndarray | None = None
    best_val_pred: np.ndarray | None = None

    def run_val(epoch_num: int) -> None:
        nonlocal best_f1, best_epoch, best_val_gold, best_val_pred
        val_loss, val_gold, val_pred, _ = evaluate(model, val_dl, device, criterion)
        val_acc = accuracy_score(val_gold, val_pred)
        val_f1m = f1_score(val_gold, val_pred, labels=list(range(len(CLASSES))),
                           average="macro", zero_division=0)
        print(f"[epoch {epoch_num} step {global_step}] "
              f"val_loss={val_loss:.4f}  acc={val_acc:.3f}  f1_macro={val_f1m:.3f}")
        wandb.log({
            "val/loss": val_loss,
            "val/accuracy": val_acc,
            "val/f1_macro": val_f1m,
            "epoch": epoch_num,
        }, step=global_step)
        if val_f1m > best_f1:
            best_f1 = val_f1m
            best_epoch = epoch_num
            best_val_gold = val_gold
            best_val_pred = val_pred
            torch.save({
                "model_name": args.model,
                "state_dict": model.state_dict(),
                "epoch": epoch_num,
                "global_step": global_step,
                "val_f1_macro": val_f1m,
                "val_accuracy": val_acc,
                "classes": CLASSES,
            }, best_path)
            print(f"  ↳ new best val_f1_macro={val_f1m:.3f}, saved → {best_path}")

    for epoch in range(args.epochs):
        freeze_backbone = epoch < args.freeze_epochs
        for p in backbone_params:
            p.requires_grad = not freeze_backbone
        if epoch == 0 or epoch == args.freeze_epochs:
            print(f"[epoch {epoch+1}] backbone {'FROZEN' if freeze_backbone else 'UNFROZEN'}")

        model.train()
        running = 0.0
        running_n = 0
        pbar = tqdm(train_dl, desc=f"epoch {epoch+1}/{args.epochs}")
        for x, y in pbar:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if use_mixcut:
                x, y_target = mixcut(x, y)
            else:
                y_target = y
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y_target)
            loss.backward()
            optimizer.step()
            scheduler.step()
            global_step += 1
            running += loss.item() * x.size(0)
            running_n += x.size(0)
            if global_step % 5 == 0:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/lr": scheduler.get_last_lr()[0],
                    "train/epoch": epoch + (running_n / max(1, len(train_dl.dataset))),
                }, step=global_step)
            pbar.set_postfix(loss=f"{running/max(1,running_n):.3f}")

        is_last_epoch = (epoch + 1) == args.epochs
        if args.eval_every_epochs > 0 and ((epoch + 1) % args.eval_every_epochs == 0 or is_last_epoch):
            run_val(epoch + 1)

    print(f"\nBest val f1_macro: {best_f1:.3f} at epoch {best_epoch}")
    print(f"Reloading best weights from {best_path} for final test metrics ...")
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    test_loss, gold_idx, pred_idx, probs = evaluate(model, test_dl, device, criterion)
    gold = [CLASSES[i] for i in gold_idx]
    preds = [CLASSES[i] for i in pred_idx]

    acc = accuracy_score(gold, preds)
    f1_macro = f1_score(gold, preds, labels=CLASSES, average="macro", zero_division=0)
    f1_weighted = f1_score(gold, preds, labels=CLASSES, average="weighted", zero_division=0)
    precision_macro = precision_score(gold, preds, labels=CLASSES, average="macro", zero_division=0)
    recall_macro = recall_score(gold, preds, labels=CLASSES, average="macro", zero_division=0)

    print(f"\nTest accuracy: {acc:.3f}")
    print(f"Macro F1: {f1_macro:.3f} | Weighted F1: {f1_weighted:.3f}")
    print("\n" + classification_report(gold, preds, labels=CLASSES, zero_division=0))
    cm = confusion_matrix(gold, preds, labels=CLASSES)
    print("Confusion matrix rows=gold cols=pred")
    print("labels:", CLASSES)
    print(cm)

    per_class = classification_report(gold, preds, labels=CLASSES, zero_division=0, output_dict=True)

    per_class_table = wandb.Table(columns=["class", "precision", "recall", "f1", "support"])
    for cls in CLASSES:
        row = per_class[cls]
        per_class_table.add_data(cls, row["precision"], row["recall"], row["f1-score"], row["support"])

    def per_class_bar(metric_key: str, display: str):
        tbl = wandb.Table(columns=["class", display],
                          data=[[cls, per_class[cls][metric_key]] for cls in CLASSES])
        return wandb.plot.bar(tbl, "class", display, title=f"Test {display} per class")

    log_payload = {
        "test/loss": test_loss,
        "test/accuracy": acc,
        "test/f1_macro": f1_macro,
        "test/f1_weighted": f1_weighted,
        "test/precision_macro": precision_macro,
        "test/recall_macro": recall_macro,
        "test/n_samples": len(gold),
        "test/best_epoch": best_epoch,
        "test/confusion_matrix": wandb.plot.confusion_matrix(
            y_true=[CLASSES.index(g) for g in gold],
            preds=[CLASSES.index(p) for p in preds],
            class_names=CLASSES,
        ),
        "test/per_class_table": per_class_table,
        "test/per_class/precision_bar": per_class_bar("precision", "precision"),
        "test/per_class/recall_bar": per_class_bar("recall", "recall"),
        "test/per_class/f1_bar": per_class_bar("f1-score", "f1"),
    }
    if best_val_gold is not None and best_val_pred is not None:
        log_payload["val/confusion_matrix_best"] = wandb.plot.confusion_matrix(
            y_true=best_val_gold.tolist(),
            preds=best_val_pred.tolist(),
            class_names=CLASSES,
        )

    mis_table = wandb.Table(columns=["filepath", "gold", "pred", "confidence", "image"])
    dataset_rows = test_dl.dataset.rows
    for i, (g, p) in enumerate(zip(gold_idx, pred_idx)):
        if g == p:
            continue
        row = dataset_rows[i]
        pil = Image.open(test_dl.dataset.processed_dir / row["filepath"]).convert("RGB")
        mis_table.add_data(
            row["filepath"], CLASSES[g], CLASSES[p],
            float(probs[i, p]), wandb.Image(pil),
        )
    log_payload["test/misclassified"] = mis_table

    wandb.log(log_payload)

    wandb.run.summary["test/accuracy"] = acc
    wandb.run.summary["test/f1_macro"] = f1_macro
    wandb.run.summary["test/f1_weighted"] = f1_weighted
    wandb.run.summary["test/precision_macro"] = precision_macro
    wandb.run.summary["test/recall_macro"] = recall_macro
    wandb.run.summary["val/f1_macro_best"] = best_f1
    wandb.run.summary["best_epoch"] = best_epoch
    for cls in CLASSES:
        wandb.run.summary[f"test/f1/{cls}"] = per_class[cls]["f1-score"]

    metrics_path = out_dir / "final_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump({
            "model": args.model,
            "best_epoch": best_epoch,
            "val_f1_macro_at_best": best_f1,
            "test_accuracy": acc,
            "test_f1_macro": f1_macro,
            "test_f1_weighted": f1_weighted,
            "test_precision_macro": precision_macro,
            "test_recall_macro": recall_macro,
            "per_class": {cls: per_class[cls] for cls in CLASSES},
            "confusion_matrix": cm.tolist(),
            "classes": CLASSES,
        }, f, indent=2)
    print(f"Wrote metrics → {metrics_path}")
    print(f"Best checkpoint → {best_path}")

    wandb.finish()


if __name__ == "__main__":
    main()
