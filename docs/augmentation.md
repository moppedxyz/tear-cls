# Augmentation

## Motivation and design constraints

AFM scans are not ordinary photographs: each pixel's colour is a
deterministic function of a scalar height measurement, mapped through the
Bruker acquisition software's lookup table. Two constraints follow.
First, any augmentation that destroys the monotone mapping between height
and colour — hue shifts, channel swaps, saturation boosts, inversions,
solarisation — also destroys the signal the model is expected to learn
and was excluded from consideration outright. Second, the features that
distinguish the five diagnostic classes are fine dendrite edges and
thin-film granularity that occupy only a handful of pixels, so even
ostensibly benign photometric operations have to be tightly bounded or
they wipe out the informative structure before the network sees it.

Within those bounds we still want to expose the model to the kinds of
variation that occur between real AFM sessions: arbitrary in-plane
rotation of the tip-sample coordinate frame, session-to-session
differences in operator-chosen contrast and gain, scan-line streaks
caused by momentary tip–surface interactions, mild focus drift, and the
possibility that a future dataset is rendered under a different
colormap than the one used here. The pipeline below is the result of
balancing these two forces.

## Pre-augmentation crop

Before any random transform is applied — and, crucially, also at
evaluation time — every image is cropped to the pure AFM data region
`(left=93, top=0, right=616, bottom=531)`, producing a near-square
523×531 patch. NanoScope's default BMP export pads the scan with white
side margins and a scale-bar strip at the bottom carrying burned-in
acquisition text. Without this crop the network learns to read the
scale bar instead of looking at the scan. The crop is treated as a
preprocessing step, not an augmentation, and cannot be disabled by an
ablation configuration.

## The augmentation pipeline

The training-time pipeline is a single Albumentations `Compose` whose
operations are applied in a fixed order. Each operation is tagged with
a short name so that ablation configurations can enable or disable it
individually. Bounds are tuned so that, on visual inspection, a
radiologist-style reviewer could not pick the augmented image out of
a set of real acquisitions.

Geometric operations (applied first):

1. `rotate` — uniform rotation in ±180°, reflective padding. AFM data
   has no canonical up direction.
2. `hflip`, `vflip` — independent horizontal and vertical flips at
   `p=0.5` each. Same justification.
3. `affine` — small translations of ±8% of image side, reflective
   padding, at `p=0.5`. No scaling or extra rotation (handled above).
4. `rrcrop` — `RandomResizedCrop` to 384×384 with scale in `[0.75,
   1.0]` and aspect ratio in `[0.9, 1.1]`, at `p=0.5`. Provides
   mild zoom variation while keeping the final tensor shape fixed.
5. `elastic` — `ElasticTransform(alpha=30, sigma=5)`, `p=0.3`. The
   parameters are deliberately at the low end of the typical range;
   stronger settings smeared dendrite edges in qualitative checks.

Photometric and scanner-realistic operations (applied after geometry,
so they compose on the remapped palette when `colormap` fires):

6. `colormap` — with `p=0.25`, convert the RGB render to luminance
   and re-render under one of five perceptually-uniform colormaps
   (viridis, plasma, inferno, magma, cividis) or pure grayscale.
   Luminance is a monotone function of height under the Bruker LUT,
   so height ordering is preserved; only the palette changes. This
   is meant to discourage the network from latching onto palette-
   specific hue priors inherited from natural-image pretraining.
7. `brightness_contrast` — `RandomBrightnessContrast` with both
   limits at 0.15, `p=0.4`.
8. `gamma` — `RandomGamma` with γ ∈ [0.85, 1.15], `p=0.3`.
9. `noise` — additive Gaussian noise with std ∈ [0.02, 0.08], mean 0,
   `p=0.3`.
10. `scanline` — a custom operator that injects 1–4 horizontal bands
    1–3 pixels tall with a ±10/255 intensity offset, `p=0.25`. This
    is the most AFM-specific operation in the set: it mimics the
    row-wise streaks that occur when the tip briefly decouples from
    the surface mid-raster.
11. `blur` — `GaussianBlur` with kernel 3–5 px, `p=0.2`.
12. `clahe` — `CLAHE` with clip limit in [1.0, 2.0] and 8×8 tile grid,
    `p=0.2`.

Augmentations are pre-materialised to disk rather than applied in the
data loader: for every training image ten augmented copies are written
once, and the training loop then samples the resulting set. This keeps
the training step deterministic with respect to its dataset indices and
makes per-configuration ablation dumps reproducible.

## Ablation study

The ablation is driven by `scripts/ablate_augmentation.py`, which loops
over named configurations and seeds, dumps a per-configuration processed
directory when needed, trains `eff_train.py` (EfficientNet-V2-S) for 30
epochs under each combination, and writes the test metrics to
`outputs/ablation_summary.csv`. The study is idempotent; partial runs
resume where they stopped.

Configurations currently in the grid:

- `full` — the complete 12-operation pipeline.
- `none` — preprocessing crop only, no augmentation.
- `geom_only` — operations 1–5.
- `photo_only` — operations 6–12.
- `loo_*` — leave-one-out ablations that disable a single photometric /
  scanner-realistic operation from the full pipeline, one per
  operation (`colormap`, `brightness_contrast`, `gamma`, `noise`,
  `scanline`, `blur`, `clahe`).

Each configuration is scheduled for three seeds (`3407`, `17`, `2026`)
so that point estimates can be reported as mean ± std over seeds. The
evaluation split is the patient-stratified held-out set defined in
`docs/splitting.md` (51 images, 9 patients, val ≡ test).

## Preliminary results (single seed)

All eleven configurations have now completed at seed `3407`; seeds
`17` and `2026` are still queued. The numbers below are therefore
**single-seed point estimates** and must be treated as directional —
on a 51-image held-out set a single misclassified image is worth
roughly two percentage points of accuracy and ~0.02 of macro-F1, so
the gaps reported here are well inside that noise band and the
multi-seed means that will go into the final paper may reorder them.
Configurations are listed in descending order of macro-F1.

| Config                       | Test acc | Macro-F1 | Δ vs `full` | Best epoch |
|------------------------------|---------:|---------:|------------:|-----------:|
| `none`                       | 0.510    | 0.542    | +0.081      | 5          |
| `loo_scanline`               | 0.510    | 0.492    | +0.031      | 10         |
| `loo_blur`                   | 0.510    | 0.489    | +0.028      | 5          |
| `loo_clahe`                  | 0.510    | 0.475    | +0.014      | 10         |
| `full`                       | 0.490    | 0.461    |  0.000      | 10         |
| `geom_only`                  | 0.510    | 0.452    | −0.009      | 10         |
| `loo_noise`                  | 0.490    | 0.436    | −0.025      | 10         |
| `loo_brightness_contrast`    | 0.471    | 0.432    | −0.029      | 10         |
| `loo_colormap`               | 0.451    | 0.422    | −0.039      | 10         |
| `loo_gamma`                  | 0.451    | 0.397    | −0.064      | 5          |
| `photo_only`                 | 0.412    | 0.324    | −0.137      | 5          |

Per-class F1 at seed 3407, same ordering:

| Config                       | diabetes | glaucoma | MS    | dry_eye | healthy |
|------------------------------|---------:|---------:|------:|--------:|--------:|
| `none`                       | 0.750    | 0.818    | 0.105 | 0.323   | 0.714   |
| `loo_scanline`               | 0.824    | 0.154    | 0.439 | 0.154   | 0.889   |
| `loo_blur`                   | 0.667    | 0.154    | 0.450 | 0.333   | 0.842   |
| `loo_clahe`                  | 0.778    | 0.000    | 0.488 | 0.286   | 0.824   |
| `full`                       | 0.526    | 0.154    | 0.488 | 0.400   | 0.737   |
| `geom_only`                  | 0.700    | 0.000    | 0.488 | 0.182   | 0.889   |
| `loo_noise`                  | 0.700    | 0.000    | 0.476 | 0.182   | 0.824   |
| `loo_brightness_contrast`    | 0.556    | 0.000    | 0.439 | 0.364   | 0.800   |
| `loo_colormap`               | 0.444    | 0.154    | 0.450 | 0.364   | 0.700   |
| `loo_gamma`                  | 0.462    | 0.154    | 0.476 | 0.167   | 0.727   |
| `photo_only`                 | 0.421    | 0.000    | 0.439 | 0.000   | 0.762   |

Three patterns are visible in the single-seed numbers and worth
flagging even though they may soften once the remaining seeds land:

1. **`none` is the best configuration on this seed.** At 189 training
   images the marginal effect of any augmentation on generalisation is
   small, and on a single-seed point estimate the augmentation block
   does net harm. The `none` per-class row also tells the cleanest
   story: glaucoma is recovered (F1 0.82) and the MS class collapses
   to 0.11, i.e. the model without augmentation overfits confidently
   to patient-specific features of one majority class rather than
   learning class-specific concepts.
2. **`photo_only` is by far the weakest configuration** at macro-F1
   0.324. Photometric operations applied without the strong geometric
   prior of rotation and flips inject more variance than useful
   signal, and drag glaucoma and `dry_eye` to F1 0.
3. **The leave-one-out scan is informative even at a single seed.**
   Reading the LOO column (Δ vs `full`):
   - **Removing `gamma` is the most damaging single change** (−0.064),
     followed by `colormap` (−0.039), `brightness_contrast` (−0.029)
     and `noise` (−0.025). These four photometric operations therefore
     appear to be the *useful* members of the photometric block.
   - **Removing `scanline` is the most beneficial single change**
     (+0.031), followed by `blur` (+0.028) and `clahe` (+0.014). These
     three operations look net-negative in the current pipeline. Of
     the three, the `scanline` finding is the most actionable: the
     custom scanner-streak operator was meant to mimic real AFM
     artefacts, but on this corpus it appears to inject distractor
     structure that the model latches onto as a class signal.
   - The contradiction between bullet 1 (`none` best) and bullet 3
     (some LOO removals beat `full`) is the natural one: the
     augmentation block contains both useful and harmful members, and
     the harmful ones (scanline, blur, clahe) currently dominate the
     net effect. A targeted "useful-only" pipeline (geometric +
     gamma + colormap + brightness/contrast + noise, dropping
     scanline / blur / clahe) is the obvious follow-up to test
     against `none` once multi-seed numbers are in.

The single-seed reordering is exactly what motivates running seeds
`17` and `2026` before publishing any of the LOO conclusions: a
0.03–0.06 swing in macro-F1 corresponds to roughly one or two
held-out images, and would be well within seed-to-seed variance for a
51-image evaluation set.

## Online vs. offline pipeline (the materialisation effect)

The numbers above all come from the **offline** pipeline: ten
augmented variants per training image are dumped to disk once by
`tearcls/augmentation.py` and the loader treats them as additional
fixed samples. Every epoch sees the same 10 augmented copies of every
image — augmentation is effectively a one-shot dataset expansion, not
a per-step regulariser. We hypothesised this was masking the true
effect of the augmentation block, since standard practice is to apply
augmentations stochastically inside `__getitem__`.

To test this, `tearcls/data.py` now supports an **online** mode
(`online_augment_names=[...]`). When set, the train split reads the
raw BMP from `source_filepath` and runs `build_train_augment` per
sample at fetch time, so every epoch sees fresh randomness; the
pre-dumped augmented PNGs in the index are ignored. Val and test are
never augmented online. `eff_train.py` exposes this as
`--online-augment <comma,separated,aug,names>` (use `''` for an
online-mode no-aug baseline). The same eleven configurations were
re-run in online mode at seed `3407`.

| Config                       | Offline F1 | Online F1 | Δ (online − offline) |
|------------------------------|-----------:|----------:|---------------------:|
| `none`                       | 0.542      | 0.559     | +0.017               |
| `geom_only`                  | 0.452      | **0.583** | +0.131               |
| `loo_gamma`                  | 0.397      | 0.573     | +0.176               |
| `loo_blur`                   | 0.489      | 0.558     | +0.069               |
| `loo_brightness_contrast`    | 0.432      | 0.540     | +0.108               |
| `loo_scanline`               | 0.492      | 0.540     | +0.048               |
| `loo_noise`                  | 0.436      | 0.537     | +0.101               |
| `full`                       | 0.461      | 0.534     | +0.073               |
| `loo_colormap`               | 0.422      | 0.531     | +0.109               |
| `photo_only`                 | 0.324      | 0.523     | +0.199               |
| `loo_clahe`                  | 0.475      | 0.507     | +0.032               |

Three things change once augmentation is applied stochastically per
epoch:

1. **Online beats offline at every single config.** The smallest gap
   is +0.017 (`none` vs. `none`, where it should be ~0 by construction
   — this residual is just stochastic batching variance) and the
   largest is +0.199 (`photo_only`). The mean gap across the eleven
   pairs is **+0.097** macro-F1. This is the strongest result in the
   ablation: the offline materialisation strategy was leaving roughly
   ten points of macro-F1 on the table across the entire grid,
   independently of which augmentations were enabled.
2. **Augmentation now has a positive effect — but only the geometric
   block.** `online_geom_only` (0.583) beats `online_none` (0.559) by
   +0.024, while `online_full` (0.534) is *below* `online_none` by
   −0.025. The geometric block (rotation, flips, affine, RRCrop,
   elastic) is the part that pulls its weight; adding the photometric
   block on top costs ~0.05 macro-F1 even with fresh randomness.
   `online_photo_only` at 0.523 is still the worst online result,
   confirming the photometric ops cannot stand alone.
3. **The LOO ranking changes.** Removing `gamma` was the most
   damaging operation in the offline grid (−0.064 vs `full`); in the
   online grid it is the *most beneficial* removal (+0.039 vs
   `online_full`), second only to `loo_blur` (+0.024 over
   `online_full`). Conversely, `loo_clahe` was a small win offline
   (+0.014) and is now a small loss (−0.027). The offline LOO
   conclusions therefore need to be discarded and read off the online
   grid instead.

The headline takeaway, single seed and all, is that
**`online_geom_only` is the current best configuration** at seed
`3407` — geometric-only augmentation, applied with fresh randomness
per epoch, beats both the offline baseline (0.461 → 0.583, +0.122)
and the online no-augmentation floor (0.559 → 0.583, +0.024). The
gain over offline-`full` (+0.122 macro-F1) is large enough to survive
seed-to-seed noise on a 51-image held-out set, although a multi-seed
run is still required before stating it as a result.

The implication for the figure-of-record is that the previous
"augmentation does not help on this corpus" reading was an artefact
of the materialisation strategy, not a property of the augmentations
themselves. The relevant question becomes whether geometric-only is
genuinely best or whether a tighter-bounded photometric subset can
beat it — to be answered once the seed-`17` / seed-`2026` runs land.

## Does the finding survive a different backbone?

To check whether the `online_geom_only` win is a property of
EfficientNet-V2-S specifically or of the task itself, the same
ablation was repeated on the much smaller `efficientnet_b0`
(5.3M parameters, 224×224 input) at seed `3407`. A five-config
subset was run — `full`, `none`, `online_full`, `online_none`,
`online_geom_only` — chosen to settle the two headline questions
(does online still beat offline; is `geom_only` still best) without
re-running the full eleven-config grid on a model whose conclusions
are not the paper's headline.

| Config              | v2_s F1 | b0 F1   | Δ (b0 − v2_s) |
|---------------------|--------:|--------:|--------------:|
| offline `full`      | 0.461   | 0.527   | +0.066        |
| offline `none`      | 0.542   | 0.632   | +0.090        |
| online `full`       | 0.534   | 0.504   | −0.030        |
| online `none`       | 0.559   | 0.632   | +0.073        |
| online `geom_only`  | **0.583** | **0.642** | +0.059       |

Two findings carry across both models:

1. **`online_geom_only` is the best configuration on both backbones.**
   On b0 it scores 0.642 vs. 0.632 for `online_none` (+0.010) and
   vs. 0.504 for `online_full` (+0.138). Geometric augmentation with
   fresh randomness is therefore the recommended pipeline regardless
   of which of the two backbones is chosen.
2. **`none` beats `full` in both the offline and online pipelines on
   both models.** The photometric block is net-negative across
   capacity, not just on v2_s. On b0 the gap is larger
   (`offline_none` − `offline_full` = +0.105) than on v2_s (+0.081),
   suggesting smaller models are even more sensitive to the
   distribution-shift cost of the photometric ops.

One finding **does not** carry across:

3. **The online-vs.-offline advantage is largely a v2_s phenomenon.**
   On b0, `online_none` (0.632) and `offline_none` (0.632) are
   identical to three decimals — as they should be, since both train
   on the same 189 originals, modulo stochastic batching. But on the
   `full` configs, b0 actually scores *worse* online (0.504) than
   offline (0.527), the opposite of v2_s. The most likely
   explanation is that b0 has less capacity to overfit to the frozen
   offline augmented copies in the first place, so the offline
   penalty that hurt v2_s does not materialise — and the additional
   per-step CPU augmentation cost in online mode gives b0 no
   regularisation benefit it can capitalise on.

A practical consequence: the **largest single result in the table is
not an augmentation effect at all**. It is the model-capacity effect.
The best v2_s configuration scores 0.583 macro-F1; the best b0
configuration scores 0.642 macro-F1, +0.059 higher. With only 33
training patients the larger model is overcapacity for the task, and
the previous decision to use v2_s as the headline architecture
should be revisited. A seed-17 / seed-2026 confirmation on b0 is now
the highest-leverage next experiment.

## Multi-seed confirmation (and a retraction)

The single-seed b0 finding above motivated a focused multi-seed
follow-up: `online_geom_only` and `online_none` on **both** v2_s and
b0 at seeds `17` and `2026`, plus `online_geom_only` on
`efficientnet_b1` (the in-between capacity point) at seed 3407, plus
b0 on the binary and three-class tasks for the progressive
decomposition story. Eleven runs total; results below.

### 5-class, multi-seed

| Backbone | Config              | Seed 3407 | Seed 17 | Seed 2026 | **Mean** | **Std** |
|----------|---------------------|----------:|--------:|----------:|---------:|--------:|
| v2_s     | `online_geom_only`  | 0.583     | 0.649   | 0.705     | **0.646** | 0.061  |
| v2_s     | `online_none`       | 0.559     | 0.558   | 0.724     | 0.614     | 0.095  |
| b0       | `online_geom_only`  | 0.642     | 0.627   | 0.587     | 0.619     | 0.029  |
| b0       | `online_none`       | 0.632     | 0.647   | 0.528     | 0.602     | 0.065  |

The single-seed story rearranges substantially:

1. **The b0 headline is retracted.** The +0.059 macro-F1 advantage of
   b0 over v2_s at seed 3407 was a lucky-seed artefact: at multi-seed
   v2_s `online_geom_only` (0.646 ± 0.061) actually beats b0
   `online_geom_only` (0.619 ± 0.029) by +0.027. v2_s remains the
   recommended backbone, and the previous suggestion to switch the
   paper's headline architecture is withdrawn.
2. **The augmentation effect is real but small.** `online_geom_only`
   beats `online_none` on both backbones (+0.032 on v2_s, +0.017 on
   b0); both gaps point the same direction across three seeds, which
   is mild Bayesian evidence in favour of the geometric block.
   Neither gap, however, is significantly larger than the
   per-configuration std — n=3 is simply not enough to reject the
   null at conventional thresholds. The honest claim is "directional
   evidence, awaiting a larger seed budget".
3. **b0 has lower variance.** Across both configs, b0's seed-to-seed
   std is roughly half v2_s's (0.029 vs. 0.061 on `online_geom_only`;
   0.065 vs. 0.095 on `online_none`). The smaller model is more
   stable across the patient-stratified held-out split, which makes
   it a better choice for hyperparameter search even if v2_s wins on
   the headline number.
4. **The seed `2026` row on v2_s `online_none` (0.724) is striking.**
   It is the single highest 5-class macro-F1 we have ever recorded.
   Without aug, on the larger backbone, on one specific seed, the
   model lands ~0.10 above its own mean. This is the clearest
   demonstration that the 51-image held-out set is too small to
   produce a stable headline number from any one run.

### Capacity is not monotone

`efficientnet_b1` (~7M params, 240px input) was added as a third
capacity point to test whether smaller-is-monotonically-better. At
seed 3407, `online_geom_only` macro-F1:

| Backbone | Params | Input | Seed 3407 macro-F1 |
|----------|-------:|------:|-------------------:|
| b0       | 5.3M   | 224   | 0.642              |
| **b1**   | **7.8M** | **240** | **0.533**       |
| v2_s     | 21M    | 384   | 0.583              |

b1 lands *below both* of its neighbours. The seed-3407 capacity
ranking is therefore non-monotone (b0 > v2_s > b1), so we cannot
explain the observed differences by a clean "smaller models suit a
smaller corpus" narrative. Most likely, what looks like a capacity
effect is dominated by per-model architecture / pretraining
particulars and the ~2pp seed noise on a 51-image set; b0's seed-3407
high was lucky and b1's was unlucky. A multi-seed confirmation on
b1 would resolve this, but is low-priority given the v2_s headline
above.

### b0 on binary and three-class

For the progressive-decomposition slide it would be informative to
report b0 numbers alongside the v2_s baselines. With
`online_geom_only` at seed 3407:

| Task     | v2_s (offline `full`) | b0 (`online_geom_only`) |
|----------|----------------------:|------------------------:|
| Binary   | 0.941 acc / 0.894 F1  | 0.863 acc / 0.815 F1    |
| 3-class  | 0.824 acc / 0.664 F1  | 0.490 acc / 0.487 F1    |

Caveat: this is not an apples-to-apples comparison — the v2_s rows
use the legacy offline `full` pipeline; the b0 rows use
`online_geom_only`. The honest reading is that **b0 does not improve
on v2_s for either of the easier tasks**; it is roughly comparable
on binary (−0.079 macro-F1) and substantially worse on three-class
(−0.177 macro-F1). Combined with the multi-seed 5-class result, the
overall conclusion is that **v2_s remains the right backbone choice
across all three task formulations**, and b0 is interesting only as a
lower-variance alternative for fast hyperparameter search.

---

## TODO: to finalise before camera-ready

- [ ] Complete seeds `17` and `2026` for all eleven configurations in
      **both** the offline and online pipelines (currently only seed
      `3407` has finished in each) and report mean ± std instead of
      single-seed point estimates.
- [ ] Paired-seed significance test (Wilcoxon signed-rank) for
      `online_geom_only` vs. `online_none` and for `online_geom_only`
      vs. `online_full`, the two contrasts that matter for the
      "is augmentation worth it / which subset" question.
- [ ] Add an `online_useful_only` configuration motivated by the
      online LOO scan — geometric + the photometric ops whose removal
      hurt online (currently only `clahe` qualifies, +0.027 below
      `online_full`) — and run across all three seeds.
- [ ] Decide whether to drop the offline pipeline entirely from the
      paper given the systematic +0.097 macro-F1 advantage of the
      online pipeline, or keep it as a contrast slide showing the
      cost of pre-materialised augmentation.
- [ ] Run more seeds (target n ≥ 5) for the four headline cells —
      v2_s `online_geom_only`, v2_s `online_none`, b0
      `online_geom_only`, b0 `online_none` — to either reject or
      confirm the +0.032 / +0.017 augmentation effect. At n=3 the
      gap is within one standard deviation.
- [ ] Re-run the v2_s binary and three-class baselines under
      `online_geom_only` so the cross-backbone progressive
      comparison is apples-to-apples (current v2_s numbers are from
      the legacy offline `full` pipeline).
- [ ] If a larger seed budget keeps v2_s `online_geom_only` ahead of
      v2_s `online_none`, run a paired-seed Wilcoxon signed-rank
      test on macro-F1 and report the p-value.
- [ ] Bar chart of macro-F1 per configuration with per-seed error bars,
      ordered by effect size.
- [ ] Qualitative figure: one source image plus one sample from each
      of the 12 operations, to make the bounds concrete for the
      reader.
- [ ] A short discussion of the `none` vs `full` finding: whether it
      survives multi-seed evaluation, and if so, whether the honest
      recommendation for future small AFM corpora is to drop the
      photometric block entirely.
- [ ] Sanity check that the 10-variants-per-image materialisation rate
      is high enough; re-run at 20 variants for the `full` and `none`
      configurations to rule out a saturation effect.
- [ ] Re-run `full` and `none` with the same number of *optimiser
      steps* rather than the same number of *epochs*, so that the
      larger effective dataset under `full` is not implicitly given a
      training-budget advantage.
