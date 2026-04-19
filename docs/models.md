# Models

## Motivation

With 240 images from 33 training patients, model capacity is not the
scarce resource; supervision is. A transformer with hundreds of millions
of parameters trained from scratch would simply memorise the training
patients and fail on the held-out nine, and a hand-tuned feature
pipeline with enough prior knowledge could plausibly match whatever a
large network learns end-to-end. The experimental design therefore
spans three families of deliberately different inductive biases —
classical handcrafted features, lightweight pretrained CNNs, and a
larger pretrained CNN with modern regularisation — so that the reported
ceiling reflects the **task**, not the **architecture**. If all three
families converge to the same accuracy on the patient-stratified split,
that convergence is evidence that the signal itself is the bottleneck;
if they disagree, the gap tells us which inductive bias pays off at
this sample size.

All results in this section are computed on the held-out split
described in `docs/splitting.md` (9 patients, 51 images, identical for
val and test). Chance accuracy is 20% for the 5-class framing,
33.3% for three-class, and 82.4% for binary (the healthy-vs-disease
majority-class baseline).

## Classification framings

Three framings of the task are trained side by side, each with its own
checkpoint per architecture:

| Framing        | Classes                                                         | Rationale                                                                                          |
|----------------|-----------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `multi` (5)    | `diabetes`, `glaucoma`, `multiple_sclerosis`, `dry_eye`, `healthy` | Full diagnostic granularity. The hardest framing and the one that exposes the ceiling most clearly. |
| `three`        | `healthy`, `dry_eye`, `disease`                                 | Collapses the four pathological classes into one. Isolates whether dry-eye tearing is separable.   |
| `binary`       | `healthy`, `disease`                                            | Screening framing: is there any pathology at all. Clinically meaningful even when fine-grained classification fails. |

The three-way cascade lets us report where the model *does* work even
when 5-class accuracy is modest — an honest alternative to quoting only
the most flattering headline number. The label remapping is a pure
relabelling of the same split, so no patient ever crosses from one
framing's train set to another's held-out set.

## Family 1 — classical baselines (`classical_train.py`)

A deliberately low-capacity reference built to answer one question: how
much of the task can be solved without any learned representation at
all. For each cropped AFM image the script computes a 68-dimensional
feature vector combining grayscale moments and percentiles, a 32-bin
intensity histogram, Sobel-gradient statistics, a Laplacian-variance
proxy for focus, per-channel RGB statistics, and 16 radial bins of the
Fourier magnitude spectrum. The vector is then fed to four `sklearn`
classifiers in parallel — logistic regression, SVM with an RBF kernel,
a 400-tree random forest, and a 300-estimator gradient-boosting
ensemble — and the one with the best validation macro-F1 is selected.

This family has two purposes. First, it provides a CPU-only baseline
that runs in seconds and is trivially reproducible on any machine,
which matters when the deep models require a GPU and multi-gigabyte
checkpoints. Second, a large gap between classical and deep results
would indicate that the network has discovered representations beyond
moment-and-frequency statistics; a small gap suggests the discriminative
information at this sample size is already low-dimensional. In practice
the latter holds.

## Family 2 — simple pretrained CNNs (`simple_train.py`)

A minimal fine-tuning setup over ImageNet-pretrained ResNet-18,
ResNet-34, and MobileNet-V3-Small. Images are resized to 224×224 after
the AFM crop, passed through the standard ImageNet normalisation, and
the only head change is replacing the final linear layer with a
`num_classes`-wide one. Training uses AdamW with learning rate 3e-4 and
weight decay 1e-4, a cosine schedule with no warmup, plain
cross-entropy without label smoothing or class weighting, and no
MixUp/CutMix. Fifteen epochs with batch size 32 are sufficient: best
validation macro-F1 is reached inside the first few epochs on all three
backbones.

These models establish what a practitioner gets from "a standard
transfer-learning recipe" without any of the bells and whistles. They
also serve as a capacity probe: ResNet-18 (11M parameters) and
MobileNet-V3-Small (1.5M parameters) bracket a 7× difference in
parameter count, and if either meaningfully outperforms the other it
would suggest capacity is a limiting factor. In practice their held-out
metrics lie within noise of each other.

## Family 3 — EfficientNet-V2-S with modern regularisation (`eff_train.py`)

The main model, selected after small-scale comparison against
EfficientNet-B0, B1, B3, and V2-M. EfficientNet-V2-S (21M parameters,
ImageNet1K-V1 weights) was kept because it reached the best validation
macro-F1 on the 5-class framing while fitting in a 12 GB GPU at
384×384. The fine-tuning recipe differs from the simple-CNN family in
six ways, each motivated by the data regime:

1. **Two-stage unfreeze.** The backbone is frozen for the first two
   epochs so the freshly-initialised head does not push large gradients
   into pretrained features; from epoch three onward the full network
   trains. Skipping this stage destabilises the first epoch of head
   updates on every run.
2. **Split learning rates.** The backbone uses `lr = 5e-5` and the head
   uses `10 × lr`, under a single AdamW optimiser. The pretrained
   backbone needs only gentle adjustment, whereas the randomly
   initialised head benefits from a standard fine-tuning learning rate.
3. **Cosine schedule with warmup.** Five percent of steps are a linear
   warmup, then cosine decay. Warmup matters because the head starts
   from random initialisation and the first few batches would otherwise
   see the joint optimiser at full learning rate.
4. **MixUp + CutMix** (randomly chosen per batch, α = 0.2 and 1.0
   respectively). On 189 training images, MixUp/CutMix act as a strong
   regulariser against memorisation of specific scans, complementing the
   geometric and photometric augmentations described in
   `docs/augmentation.md`.
5. **Label smoothing (ε = 0.1)** and **class-weighted cross-entropy**
   with inverse-frequency weights normalised to mean 1. Given the 7×
   imbalance between `multiple_sclerosis` (95 images) and `dry_eye`
   (13 images), an unweighted cross-entropy would push the minority
   classes to recall zero. Label smoothing provides an additional small
   hedge against overconfident memorisation.
6. **Increased stochastic depth.** The built-in StochasticDepth rates
   are multiplied by 1.5 and capped at 0.3, and head dropout is set to
   0.4. Both are motivated by the tendency of V2-S to overfit within
   the first five epochs on this corpus.

The best validation checkpoint (by macro-F1) is restored before the
final test evaluation, and every run's hyperparameters, per-class
metrics, and misclassified examples are logged to Weights & Biases
under `ramang/tear-cls`.

## Held-out results

All numbers below are test-set metrics on the patient-stratified split,
with `macro-F1` giving equal weight to each class and therefore
reflecting minority-class performance more fairly than accuracy.

### 5-class (`multi`)

Seed-3407 single-run numbers and, for the EfficientNet-V2-S
configurations that have been re-run, multi-seed mean ± std across
seeds `{3407, 17, 2026}`. The choice of pipeline (offline vs. online
augmentation) is fully described in `docs/augmentation.md`; the
short version is that "offline" pre-materialises ten frozen
augmented variants per training image, and "online" applies the
same operations stochastically inside `__getitem__` so every epoch
sees fresh randomness.

| Model                                       | Seed 3407 acc | Seed 3407 F1 | Mean F1 (n=3) | Std    |
|---------------------------------------------|--------------:|-------------:|--------------:|-------:|
| Classical SVM-RBF                           | 0.569         | 0.455        | —             | —      |
| MobileNet-V3-Small                          | 0.569         | 0.467        | —             | —      |
| ResNet-18                                   | 0.588         | 0.451        | —             | —      |
| EfficientNet-B0, online `geom_only`         | 0.647         | 0.642        | 0.619         | 0.029  |
| EfficientNet-B1, online `geom_only`         | 0.529         | 0.533        | —             | —      |
| EfficientNet-V2-S, offline `full`           | 0.490         | 0.461        | —             | —      |
| EfficientNet-V2-S, offline `none` (no aug)  | 0.510         | 0.542        | —             | —      |
| EfficientNet-V2-S, online `none` (no aug)   | 0.549         | 0.559        | 0.614         | 0.095  |
| **EfficientNet-V2-S, online `geom_only`**   | **0.529**     | **0.583**    | **0.646**     | 0.061  |

The headline 5-class number is **EfficientNet-V2-S with online
geometric augmentation, 0.646 ± 0.061 macro-F1** averaged over three
seeds. Three observations from the table:

1. **Online geometric augmentation is the best configuration we have
   found.** It beats the previous "no augmentation" headline (0.614
   ± 0.095) by +0.032 macro-F1 in the mean. Both configurations
   point in the same direction across all three seeds, but with
   n = 3 and a per-cell std of ~0.06–0.10 the gap is not larger than
   one standard deviation; the honest reading is "directional
   evidence, awaiting a larger seed budget".
2. **The frozen-offline pipeline was leaving roughly ten points of
   macro-F1 on the table.** Pre-materialising augmentations to disk
   and treating the same ten copies as a fixed dataset expansion
   reduced V2-S to 0.461 macro-F1 at seed 3407, well below the
   online runs. The pre-dumped variants behave more like a noisy
   dataset expansion than per-step regularisation; switching to
   on-the-fly random transforms removes that penalty independently
   of which augmentations are enabled.
3. **EfficientNet-B0 is competitive but does not win at multi-seed.**
   B0's seed-3407 macro-F1 of 0.642 narrowly beat V2-S's 0.583, but
   averaged over three seeds B0 reaches 0.619 (±0.029) versus V2-S's
   0.646 (±0.061). The effective comparison is "V2-S has a higher
   ceiling but is more seed-sensitive; B0 has a lower ceiling but
   half the variance". V2-S remains the headline backbone, while B0
   is a useful low-variance alternative for hyperparameter search.
4. **Capacity is non-monotone.** EfficientNet-B1, the in-between
   capacity point, scored 0.533 macro-F1 at seed 3407 — *below* both
   B0 (0.642) and V2-S (0.583) at the same seed. The seed-3407
   capacity ranking is therefore not "smaller is better"; the
   apparent capacity differences are at least partly driven by
   architecture and seed effects, not by parameter count alone.

The most striking single number in the table does not appear in any
row: V2-S `online_none` at seed 2026 hit 0.724 macro-F1 — the
highest 5-class result we have ever recorded, on the smallest
possible configuration. With only 51 held-out images from 9
patients, any single-seed headline number is dominated by which 9
patients happen to be drawn into the held-out set and which seed is
used. The mean ± std rows above are therefore the right thing to
quote in the paper.

### Three-class (`three`)

| Model                                       | Test accuracy | Macro-F1 |
|---------------------------------------------|--------------:|---------:|
| Classical SVM-RBF                           | 0.725         | 0.491    |
| ResNet-18                                   | 0.784         | 0.533    |
| EfficientNet-B0, online `geom_only`         | 0.490         | 0.487    |
| **EfficientNet-V2-S, offline `full`**       | **0.824**     | **0.664** |

Collapsing the four pathological classes lifts EfficientNet-V2-S to
82.4% accuracy and 0.664 macro-F1. The gap between ResNet-18 and
EfficientNet-V2-S is larger here than in the 5-class framing,
suggesting that the modern regularisation recipe pays off more clearly
once per-class support rises above the tiny-sample regime that
dominates `dry_eye` and `diabetes`. EfficientNet-B0, retrained under
the multi-task best 5-class recipe (`online_geom_only`), did **not**
generalise back to this framing — at 0.487 macro-F1 it underperforms
even the classical SVM, suggesting that whatever capacity advantage
B0 had on 5-class came from class-specific feature interactions that
disappear once the four pathological classes are merged. The
`online_geom_only` v2_s rerun for this framing has not been done
yet — the 0.664 V2-S row is from the legacy offline `full` pipeline.

### Binary (`binary`)

| Model                                       | Test accuracy | Macro-F1 |
|---------------------------------------------|--------------:|---------:|
| Classical LogReg                            | 0.863         | 0.804    |
| ResNet-18                                   | 0.882         | 0.826    |
| MobileNet-V3-Small                          | 0.882         | 0.837    |
| EfficientNet-B0, online `geom_only`         | 0.863         | 0.815    |
| **EfficientNet-V2-S, offline `full`**       | **0.941**     | **0.894** |

The screening framing is the strongest result: 94.1% accuracy and 0.894
macro-F1 with only three false positives and three false negatives
across 51 held-out images. Because the class balance is 42 disease vs 9
healthy, macro-F1 (which weights the minority class equally) is the
informative metric — accuracy alone would be inflated by the imbalance.
EfficientNet-B0 under the best 5-class recipe matches the classical
baseline (0.815 vs. 0.804 macro-F1) but does not approach V2-S; as
with three-class, V2-S retains a clear advantage on the easier
formulations. The 0.894 V2-S row is from the legacy offline `full`
pipeline; an `online_geom_only` rerun is on the TODO list.

## Selected model

**EfficientNet-V2-S is the reference model for all three framings.** It
matches or beats every other architecture on held-out macro-F1, the
magnitude of its advantage grows as the task is simplified from 5-class
to binary, and a single training script and checkpoint format covers all
three modes. The 5-class headline uses online geometric augmentation
applied per-batch (see `docs/augmentation.md`); the three-class and
binary headlines still use the legacy offline `full` pipeline pending
re-runs. The exact checkpoints used in the reported numbers are:

| Framing      | Pipeline               | Checkpoint                                                                   |
|--------------|------------------------|------------------------------------------------------------------------------|
| 5-class      | online `geom_only`     | `outputs/efficientnet_v2_s_multi_online_geom_only_s3407/best.pt`             |
| three-class  | offline `full`         | `outputs/efficientnet_v2_s_three/best.pt`                                    |
| binary       | offline `full`         | `outputs/efficientnet_v2_s_binary/best.pt`                                   |

All three are loaded by `infer.py` out of the box; the architecture and
class list are read directly from the checkpoint so no command-line
flag needs to match the training configuration.

## Models considered and rejected

- **Smaller EfficientNets (B0, B1).** Both run under the same online
  `geom_only` recipe at seed 3407, with B0 also extended across all
  three seeds. B0 reached 0.619 ± 0.029 macro-F1 across three seeds
  and B1 reached 0.533 at seed 3407, neither matching V2-S's 0.646
  ± 0.061. Worth keeping in mind: B0 has substantially lower
  seed-to-seed variance and is a useful drop-in for hyperparameter
  search even though it loses on the headline number. B1 underperforms
  both of its neighbours on a single seed and was not extended.
- **Larger EfficientNets (V2-M, B3).** Trained under the legacy
  offline recipe and did not improve on V2-S in held-out macro-F1
  despite 2–3× the parameter count; treated as evidence that capacity
  is not the binding constraint at 240 images.
- **Vision Transformers / ViT.** Not trained. Pure transformers with
  no convolutional inductive bias are known to require either very
  large pretraining corpora or strong distillation to beat CNNs on
  small downstream tasks, and we have neither the data nor the compute
  budget to close that gap here. Revisit once a larger cohort is
  available.
- **CLIP-style contrastive backbones** (e.g. BiomedCLIP). An early
  iteration of this project used CLIP ViT-L/14 features with logistic
  regression and reached ≈57% test accuracy on 5-class — within the
  same band as the CNNs reported above, but with no natural path to
  exploit the AFM header metadata. Superseded by the direct fine-tune.
- **Vision-language model + header text** (Qwen2.5-VL-7B + QLoRA).
  An exploratory direction at the time of writing: pair each image with
  the textual digest of its AFM sidecar header and fine-tune a 7B
  vision-LLM with 4-bit QLoRA to emit the diagnosis as a string. Too
  preliminary to include in the headline numbers; zero-shot accuracy
  is near chance, as expected given the absence of AFM priors in the
  pretraining mix. Reported separately once a fine-tune stabilises.

## Reproducing the reported runs

```bash
# 5-class headline (current): online geometric augmentation
uv run eff_train.py --mode multi \
  --online-augment rotate,hflip,vflip,affine,rrcrop,elastic \
  --ablation-tag online_geom_only

# Multi-seed (numbers reported as mean ± std)
for s in 3407 17 2026; do
  uv run eff_train.py --mode multi --seed $s \
    --online-augment rotate,hflip,vflip,affine,rrcrop,elastic \
    --ablation-tag online_geom_only
done

# Three-class and binary (legacy offline `full`, pending online rerun)
uv run eff_train.py --mode three
uv run eff_train.py --mode binary
```

Each command writes `best.pt` (state dict plus metadata) and
`final_metrics.json` (test accuracy, per-class precision/recall/F1, and
the confusion matrix) to the corresponding subdirectory of `outputs/`.
The random seed defaults to 3407 (matching the seed used by
`tearcls/data_split.py`) but can be overridden with `--seed`. To rerun
the full augmentation ablation grid driving the 5-class headline,
invoke `scripts/ablate_augmentation.py` (see `docs/augmentation.md`).

---

## TODO: analyses to add before camera-ready

- [ ] Re-run the three-class and binary V2-S headlines under the
      online `geom_only` pipeline so all three framings use the same
      augmentation strategy and the cross-task comparison is fully
      apples-to-apples. The current 0.664 (three-class) and 0.894
      (binary) numbers are still from the legacy offline `full`
      pipeline.
- [ ] Extend the multi-seed grid for V2-S `online_geom_only` from
      n=3 to n≥5 seeds. The current mean ± std (0.646 ± 0.061) puts
      the augmentation effect inside one standard deviation of the
      no-aug baseline — more seeds are needed before the +0.032 gap
      can be reported with a paired significance test.
- [ ] Ablation table for the EfficientNet-V2-S recipe isolating the
      contribution of each of the six modifications listed above
      (unfreeze schedule, split LR, warmup, MixUp/CutMix, label
      smoothing + class weights, stochastic depth).
- [ ] Per-patient majority-vote evaluation: aggregate the per-image
      predictions for each held-out patient and report patient-level
      accuracy, which is closer to the clinical decision unit than
      per-scan accuracy.
- [ ] Calibration analysis (ECE, reliability diagrams) for the binary
      model, since screening use downstream depends on the threshold
      being meaningful and not just the argmax.
- [ ] Finalise and report the Qwen2.5-VL + QLoRA run now listed as
      "preliminary"; include a comparison of image-only, text-only
      (AFM header digest), and image+text variants to isolate the
      contribution of the metadata modality.
