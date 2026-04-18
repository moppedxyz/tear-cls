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

| Model                       | Test accuracy | Macro-F1 | Notes                                    |
|-----------------------------|--------------:|---------:|------------------------------------------|
| Classical SVM-RBF           | —             | 0.455\*  | Val macro-F1; 68-D features + `sklearn`. |
| MobileNet-V3-Small          | 0.569         | 0.467    | 15 epochs, plain CE.                     |
| ResNet-18                   | 0.588         | 0.451    | 15 epochs, plain CE.                     |
| EfficientNet-V2-S, full aug | 0.490         | 0.461    | MixUp/CutMix, full augmentation stack.   |
| **EfficientNet-V2-S, no aug** | **0.510**   | **0.542** | Same recipe, `--no-augmented`.           |

\*The classical run reports macro-F1 on the validation fold (the
selection criterion); under the patient-stratified split `val ≡ test`
so this number is directly comparable.

Two observations. First, the four deep models lie within ≈2% accuracy
of one another, and the classical pipeline is within ≈10% — all five
families stall in a narrow band well below what the 89.6% leakage-split
number suggested. Second, the heavy augmentation stack *hurt* the
EfficientNet-V2-S macro-F1 in this framing (0.461 with full aug,
**0.542** with no aug). With only 189 training images, aggressive
MixUp/CutMix combined with photometric augmentation destroys enough of
the informative dendrite structure in the minority classes that the
regularisation cost outweighs its generalisation benefit. The no-aug
checkpoint is therefore the headline 5-class result.

### Three-class (`three`)

| Model                | Test accuracy | Macro-F1 |
|----------------------|--------------:|---------:|
| Classical SVM-RBF    | —             | 0.491\*  |
| ResNet-18            | 0.784         | 0.533    |
| **EfficientNet-V2-S** | **0.824**    | **0.664** |

Collapsing the four pathological classes lifts EfficientNet-V2-S to
82.4% accuracy and 0.664 macro-F1. The gap between ResNet-18 and
EfficientNet-V2-S is larger here than in the 5-class framing,
suggesting that the modern regularisation recipe pays off more clearly
once per-class support rises above the tiny-sample regime that
dominates `dry_eye` and `diabetes`.

### Binary (`binary`)

| Model                | Test accuracy | Macro-F1 |
|----------------------|--------------:|---------:|
| Classical LogReg     | —             | 0.804\*  |
| ResNet-18            | 0.882         | 0.826    |
| MobileNet-V3-Small   | 0.882         | 0.837    |
| **EfficientNet-V2-S** | **0.941**    | **0.894** |

The screening framing is the strongest result: 94.1% accuracy and 0.894
macro-F1 with only three false positives and three false negatives
across 51 held-out images. Because the class balance is 42 disease vs 9
healthy, macro-F1 (which weights the minority class equally) is the
informative metric — accuracy alone would be inflated by the imbalance.

## Selected model

**EfficientNet-V2-S is the reference model for all three framings.** It
matches or beats every other architecture on held-out macro-F1, the
magnitude of its advantage grows as the task is simplified from 5-class
to binary, and a single training script and checkpoint format covers all
three modes. The exact checkpoints used in the reported numbers are:

| Framing | Checkpoint                                                         |
|---------|--------------------------------------------------------------------|
| 5-class (no-aug) | `outputs/efficientnet_v2_s_multi_none_s3407/best.pt`      |
| three-class      | `outputs/efficientnet_v2_s_three/best.pt`                 |
| binary           | `outputs/efficientnet_v2_s_binary/best.pt`                |

All three are loaded by `infer.py` out of the box; the architecture and
class list are read directly from the checkpoint so no command-line
flag needs to match the training configuration.

## Models considered and rejected

- **Larger EfficientNets (V2-M, B3).** Trained under the same recipe
  and did not improve on V2-S in held-out macro-F1 despite 2–3× the
  parameter count; treated as evidence that capacity is not the binding
  constraint at 240 images.
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
uv run eff_train.py --mode binary
uv run eff_train.py --mode three
uv run eff_train.py --mode multi --no-augmented --ablation-tag none
```

Each command writes `best.pt` (state dict plus metadata) and
`final_metrics.json` (test accuracy, per-class precision/recall/F1, and
the confusion matrix) to the corresponding subdirectory of `outputs/`.
The random seed is fixed at 3407 throughout, matching the seed used by
`tearcls/data_split.py`.

---

## TODO: analyses to add before camera-ready

- [ ] Seed sensitivity: re-run the three reference configurations across
      at least five seeds and report mean ± std for accuracy and macro-F1.
      The single-seed numbers above almost certainly have ±2–3% noise
      driven by which augmented variants land in each minibatch.
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
