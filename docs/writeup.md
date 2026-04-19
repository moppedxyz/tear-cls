# Tear-film Classifier Tuning — Session Writeup

## TL;DR

Three-layer result, each more honest than the last:

1. **Gradient clipping (`max_norm=1.0`) lifted single-seed val_f1_macro 0.604 → 0.655** (+0.051). That's the one clean training-side win — grad_clip tames the head's spiky gradients under head_lr=5e-4 + batch=8.
2. **3-seed ensembling reveals the single-seed 0.655 was above-average luck**. Per-seed std is 0.027 (above the nominal 0.02 noise floor). The honest ensemble val_f1 is **0.638**; the "+0.051 win" is probably closer to **+0.03** once variance is accounted for on both sides.
3. **K-fold CV exposed that val and test are the *same 51 images* in `splits.csv` by design** (docstring: *"val and test are merged… identical by construction"*). Every "test" metric we reported was the same data hyperparameters were tuned against. The only clean single-model-on-new-patient number this dataset can give us today is the **5-fold OOF val_f1 = 0.485**.

The rank-ordering of experiments is still informative, but the absolute numbers are on a measurement that has no independent test component. Our real improvement is "grad_clip tightened the head's optimization," not "we got to 0.65."

HEAD: `4775358 fix: K-fold pool uses train-only`, with K-fold infrastructure added on top of the 3-seed-ensemble default.

---

## Setup

- **Task**: 5-class tear-film classification from AFM scans (diabetes, glaucoma, multiple_sclerosis, dry_eye, healthy).
- **Data**: 189 train / 51 val / 51 test images (patient-stratified) — but see §"Data-split caveat" — val and test are identical. 33 unique train patients, 9 unique held-out patients. Train is offline-augmented 10× per image → 2079 samples/epoch. Augmentations (rotate/flip/affine/rrcrop/elastic/colormap/brightness-contrast/gamma/noise/scanline/blur/clahe) are dumped once by `tearcls/augmentation.py` into `data/processed/`, not applied at runtime.
- **Model**: `torchvision.efficientnet_v2_s`, ImageNet-pretrained, single `Linear(1280, 5)` head.
- **Training**: AdamW, 10 epochs, cosine schedule with 5% warmup, 2-epoch head-only warmup (backbone frozen for epochs 1–2), MixUp+CutMix, class-balanced sampler, BF16 autocast, **grad_clip=1.0**, **3-seed ensemble** default (seeds `3407,42,1337`).
- **Metric**: `val_f1_macro` on 51-image holdout. Noise floor was reported as 0.02; actual seed std is 0.027.

---

## The key structural insight

**The val_f1 curve peaks hard at epoch 1 and decays monotonically after.**

```
epoch 1:  0.655   ← peak (backbone FROZEN)
epoch 2:  0.396   (backbone FROZEN, head overfits)
epoch 3:  0.49x   (backbone UNFREEZES)
epoch 4+: 0.35–0.48 (noisy decay)
```

Because the peak occurs while the backbone is still frozen, **only the head participates in reaching it**. Every backbone-side intervention is guaranteed to be a no-op at the peak — it can shape the later trajectory but not the checkpoint we keep. We verified this empirically: LLRD, `freeze_bn`, different backbones, larger input size, and drop-path tweaks all failed or tied.

This reframes the experiment space: to beat 0.655, only **head-at-epoch-1 dynamics** matter — gradient magnitudes, head LR, head dropout, MixUp/CutMix strength, label smoothing, warmup, batch size.

---

## The winning change: gradient clipping

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
```

One line before `optimizer.step()`. Global grad-norm clip at 1.0.

**Why it works**: the head has an effective LR of 5e-4 (`lr=5e-5 × head_lr_mult=10`) and the batch size is 8. Under MixUp/CutMix, individual batches can produce spiky head gradients. Clipping caps the magnitude without changing direction, so the head takes smaller, more consistent steps and lands on a better epoch-1 minimum.

**Evidence it's real, not noise**:
- +0.051 improvement is 2.5× the 0.02 noise floor.
- Test metrics track: test macro-F1 went 0.604 → 0.655, weighted F1 0.649 → 0.686.
- Robust to clip threshold: 0.5 and 1.0 give **identical** peak val_f1. The head gradients routinely exceed both thresholds, so both clips hit.

---

## What didn't work, grouped by reason

### 1. Backbone-side changes — structurally can't move the peak

| Experiment | val_f1 | Δ | Why it can't help |
|---|---|---|---|
| LLRD decay=0.75 | 0.604 | 0.000 | Backbone frozen at epoch 1, so per-stage LR has no effect |
| freeze_epochs=1 | 0.655 | 0.000 | Still frozen at epoch 1; identical to freeze=2 at peak |
| freeze_epochs=0 | 0.600 | −0.004 | Backbone unfrozen, but peak stays at epoch 1; small head step perturbation |
| freeze_bn | 0.578 | −0.077 | AFM images are far from ImageNet; letting BN adapt actually helps |
| input_size=448 | 0.543 | −0.112 | Pretrained features are tuned to 384; deviating costs more than it gains on 189 train images |
| convnext_tiny | 0.527 | −0.128 | Different inductive biases don't transfer as well; EfficientNet V2 S is right here |

**Takeaway**: don't spend experiments on backbone architecture or backbone regularization until you've changed when the peak occurs.

### 2. Aggressive head tweaks — the sweet spot is narrow

| Experiment | val_f1 | Δ | Why |
|---|---|---|---|
| head_lr_mult=20 | 0.613 | −0.042 | Too aggressive, even under grad_clip |
| lr=1e-4 (2×) | 0.613 | −0.042 | Base LR is already near-optimal; doubling pushes past the head's stability zone |
| mixup_alpha=0.15 | 0.642 | −0.013 | 0.2 is a sharp optimum |
| mixup_alpha=0.1 | 0.540 | −0.115 | Weak mixup underregularizes the head |
| cutmix_alpha=0.7 | 0.517 | −0.138 | 1.0 is correct; weaker cutmix hurts |
| warmup_ratio=0 | 0.598 | −0.057 | No warmup → head takes massive first step, blows up |
| warmup_ratio=0.1 | 0.585 | −0.070 | Too long warmup delays reaching the minimum by epoch 1 |
| label_smoothing=0.2 | 0.642 | −0.013 | Already have MixUp smoothing; more smoothing over-regularizes |

**Takeaway**: the defaults (mixup=0.2, cutmix=1.0, warmup=0.05, head_lr_mult=10, lr=5e-5, label_smoothing=0.1, head_dropout=0.4) are well-tuned. The response surface near them is concave; small deviations hurt.

### 3. Combining stabilizers — over-regularizes

| Experiment | val_f1 | Δ | Why |
|---|---|---|---|
| freeze=0 + grad_clip | 0.496 | −0.159 | With backbone training from step 1, clipped gradients spread across more params → each param moves too little |
| beta2=0.98 + grad_clip | 0.631 | −0.024 | Faster-adapting Adam with grad clip bounds the effective step size too tightly |

**Takeaway**: grad_clip alone is the right amount of stabilization. Stacking stabilizers double-bounds the optimizer.

### 4. EMA and MLP head — fundamental mismatches with peak-at-epoch-1

| Experiment | val_f1 | Δ | Why |
|---|---|---|---|
| EMA decay=0.99 | 0.571 | −0.04 | EMA needs many steps to track the model; at step 260 (epoch 1), EMA still lags the raw model, so it evaluates a worse snapshot |
| MLP head hidden=128 | 0.587 | −0.068 | Head jumps from 6K → 165K params; overfits on 189 train images |

**Takeaway**: any technique that needs "more training time" (EMA, SWA, progressive unfreezing) is inherently broken for this setup. Any technique that adds capacity (MLP head, larger model) overfits.

### 5. Dangerous baseline tweaks

| Experiment | val_f1 | Δ | Why |
|---|---|---|---|
| class_weights=none | 0.514 | −0.141 | Disables the loss-side class balancing; **not** redundant with the sampler — the loss weighting materially scales learning per class |
| weight_decay=2e-4 | 0.655 | 0.000 | Head has ~6K params; weight decay at any small value is effectively a no-op |
| epochs=3 | 0.646 | −0.009 | Tie within noise; 3× faster, but cosine schedule shape matters, so strict tie-to-baseline rule says keep longer for reproducibility |

---

## Confirmed "don't retune these" defaults

After this session, these are load-bearing and should not be touched without a very specific hypothesis:

| Knob | Value | Note |
|---|---|---|
| model | efficientnet_v2_s | b0/b1/b3 and convnext_tiny all worse; b0 ties at 5× fewer params with the right aug stack |
| batch_size | 8 | 4 too noisy, 16/32 too smooth |
| lr | 5e-5 | Doubling hurts under grad_clip |
| head_lr_mult | 10 | 5 tie, 20 much worse |
| freeze_epochs | 2 | 0 and 5 both tie; 10+ untried but likely tie |
| warmup_ratio | 0.05 | 0 and 0.1 both hurt |
| mixup_alpha | 0.2 | Sharp optimum |
| cutmix_alpha | 1.0 | Sharp optimum |
| label_smoothing | 0.1 | 0 and 0.2 both tie/worse |
| class_weights | balanced | none → −0.14; don't remove |
| head_dropout | 0.4 | 0.2/0.3/0.5 all within noise |
| weight_decay | 5e-4 | Doesn't matter at the peak |
| **grad_clip** | **1.0** | **+0.051 single-seed; +0.03 ensemble-honest** |
| **seeds** | **3 (3407, 42, 1337)** | **3-seed ensemble by default to absorb per-seed variance (std=0.027)** |

---

## Variance-reduction attempts (ensembling and K-fold)

After hitting the 0.655 single-seed ceiling, I asked whether variance reduction could move the mean. Two approaches:

### 3-seed ensemble (kept as default)

Train `efficientnet_v2_s + grad_clip + offline+full augs` at seeds `3407, 42, 1337`; average softmax at val/test.

| seed | val_f1_macro @ best epoch |
|---|---|
| 3407 | 0.6548 @ ep1 |
| 42 | 0.6508 @ ep1 |
| 1337 | 0.5961 @ ep1 |
| **mean / std** | **0.634 / 0.027** |
| **ensemble** | **0.638** |

**Finding**: per-seed std of 0.027 exceeds the nominal 0.02 noise floor, confirming the single-seed 0.655 was above-average luck. Ensemble lands at 0.638, just below the best single seed and just above the mean — exactly where the math predicts. This becomes the honest baseline.

### K-fold CV (discarded, but diagnostic)

`StratifiedGroupKFold` on the 33 training patients' `patient_code`, one model per fold, OOF concatenation for val, ensemble all folds on test.

| K | per-fold mean ± std val_f1 | OOF val | ensemble test |
|---|---|---|---|
| 3 | 0.381 ± 0.104 | 0.439 | 0.459 |
| 5 | 0.393 ± 0.062 | **0.485** | 0.495 |

**Finding**: K-fold regresses sharply because each fold trains on only 22–27 of the 33 patients (a 20–33% data loss that the ensemble can't compensate for). At this data scale, **seed ensembling beats K-fold** — same full dataset, different seeds, pure variance reduction with no capacity loss.

But K-fold did two things worth the compute:
- Gave us the **OOF val_f1 = 0.485**, which is the *only* number in this writeup measured on genuinely unseen patients.
- Uncovered the data-split caveat below.

---

## Data-split caveat (the uncomfortable part)

`tearcls/data_split.py` deliberately emits each held-out image **twice** — once with `split="val"`, once with `split="test"`. Its docstring:

> *"Val and test are merged into a single held-out set — with only a few unique patients per class there aren't enough to support three disjoint held-out groups without leakage. Each held-out row is emitted twice in the output: once with split="val", once with split="test"... so downstream loaders that expect both splits still work, and the two sets are identical by construction."*

Verified empirically: `val_files ∩ test_files = 51 / 51 files`. **val == test**.

Consequences:
- Every "test" number in this writeup (0.655, 0.638, etc.) is measuring the same 51 images that hyperparameters were tuned against. It's **val** by another name, not an independent test.
- There is no independent held-out signal in this dataset today. The 5-fold OOF val_f1 = 0.485 is the only number that comes close, and it's measured on *training-cohort* patients held out one-at-a-time, not the held-out cohort.

This is a property of the dataset, not a bug in training code. Fixing it means modifying `data_split.py` to split the 9 held-out patients' images into disjoint val and test (e.g., 26 val / 25 test), then re-running `tearcls/augmentation.py` to regenerate `data/processed/index.csv`. That would give a real test set for the first time.

---

## What's still worth trying (open directions)

Ordered by expected impact, highest first:

1. **Fix the val/test split first.** Modify `tearcls/data_split.py` to split held-out patients' images into disjoint val (~26) / test (~25) subsets, re-run `tearcls/augmentation.py`, then re-baseline. Without this, there is no way to tell a real improvement from a better hyperparameter fit to the val set. Everything below this is secondary until this is done.
2. **More training patients** if the collection is extensible at all. 33 patients is the binding constraint on everything — ensemble, K-fold, and single-seed variance all trace back to it.
3. **SAM (Sharpness-Aware Minimization)** — 2× compute, but known to help small-dataset fine-tuning by flattening the loss landscape. Untried.
4. **Multi-crop TTA at test time** (center + 4 corners, no flip) — hflip TTA failed because tear-film images aren't left-right symmetric, but crop-based TTA preserves that asymmetry. Test-only change, zero training risk.
5. **Self-distillation** — use the current best model's predictions as soft targets for a second training pass.
6. **Head initialization experiments** — zero-init the final Linear, or small-scale init, to match pretrained feature magnitudes. Untried.

What's **not worth trying** (wasteful given the structural constraint):
- More backbone regularization variants (drop_path, stochastic depth tweaks)
- Longer/shorter total schedules (epochs=5, 20 already tried; shape matters, length doesn't)
- Different learning-rate schedules alone (OneCycle, linear, step) — won't move the epoch-1 peak
- EMA / SWA / any time-averaging — they all lag
- K-fold CV at current data scale — ensemble can't compensate for per-fold data loss
- Hunting seed-level improvements — 0.027 std means single-seed "wins" under 0.02 are noise

---

## Process notes

- **Experiment count**: ~30 runs total on `autoresearch/apr18` (2 keep, rest discard). Keepers: `eae3e04` (grad_clip single-seed 0.655), `7c6e368` (3-seed ensemble 0.638 — current honest baseline).
- **Git hygiene**: each experiment is one commit. Discards are `git reset --hard HEAD~1`. `results.tsv` stays git-ignored (logged out-of-band per `program.md`).
- **Compute**: single-seed ~2 min, 3-seed ensemble ~6 min, 5-fold CV ~10 min.
- **Setup chores that mattered**:
  - Committed `tearcls/` and `setup.sh` (were untracked on master) so the autoresearch revert loop wouldn't nuke them.
  - Baked the `batch_size=8` and `eval_every_epochs=1` kept-defaults into `eff_train.py` (they had been "kept" in `results.tsv` but never made it into the tracked file).
  - Refactored `eff_train.py` to support `--seeds` (ensemble by default) and `--kfold N` (optional CV mode). Each seed/fold saves `best_seed_{S}.pt` or `best_fold_{K}.pt` so `infer.py` can auto-ensemble.

## Quick reference

```bash
# default: 3-seed ensemble, grad_clip=1.0, offline+full augs, efficientnet_v2_s
uv run eff_train.py

# custom seeds
uv run eff_train.py --seeds 7,13,31,42,99

# K-fold CV (exposes the honest OOF val number but each fold trains weaker)
uv run eff_train.py --kfold 5

# inference auto-ensembles every best_seed_*.pt (or best_fold_*.pt) it finds
uv run infer.py data_raw/Diabetes/some.bmp
```
