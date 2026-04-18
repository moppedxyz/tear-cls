# Data splitting

## Motivation

The corpus described in the previous section contains 240 scans but only on
the order of a few dozen unique patients, because each clinic session
typically produced several AFM scans of the same tear film. A naive
class-stratified split that shuffles images independently would place scans
from the same patient into both the training and the evaluation set. In an
early iteration of this work exactly that mistake was made, and it yielded
an apparently strong 89.6% test accuracy that collapsed as soon as the
split was redone at the patient level. We therefore commit, for every
experiment reported in this paper, to a strictly **patient-level split**:
no patient identifier ever appears in more than one of `{train, held-out}`.

## Patient identity recovery

Patient identity is not stored inside the BMP or the AFM sidecar. It is
recovered from the filename stem by cutting on the first numeric
acquisition-index segment. For example, `37_DM.010_1.bmp` resolves to
patient `37_DM`, and `DM_01.03.2024_LO.001_1.bmp`, in which the date is
embedded in the stem, resolves to `DM_01.03.2024_LO`. The regular
expression that performs this cut is intentionally conservative: it only
strips a `.` followed by digits (the vendor's scan-index separator) and
keeps any dots belonging to an encoded date. All scans whose stems reduce
to the same string are treated as originating from the same session of the
same patient and are kept together in whichever split they land.

## Splitting procedure

Splitting is performed independently per class so that the patient-level
80/20 ratio holds within every diagnostic group, not merely in aggregate.
Concretely, for each of the five classes:

1. All BMP filenames belonging to the class are collected and their
   patient codes recovered as above.
2. The set of unique patient codes is shuffled with a fixed seed
   (`SEED = 3407`) using Python's `random.Random` so that the split is
   reproducible across machines and runs.
3. A held-out fraction `EVAL_FRAC = 0.2` of patients is taken from the
   front of the shuffled list, with a floor of one patient; the remainder
   go to training.
4. Every image belonging to a training patient is assigned `split=train`;
   every image belonging to a held-out patient is assigned both
   `split=val` and `split=test` (one physical row per split label in the
   output CSV).

The val and test sets are therefore **identical by construction**. The
alternative — carving a third disjoint group of patients — is not
viable at this corpus size, since several classes have only a handful of
unique patients to begin with, and a three-way split would either produce
empty held-out groups or force leakage. Emitting the same rows under two
split labels keeps downstream code that expects both a validation and a
test partition working without modification, while making the equality
explicit to any reader of the CSV.

When a class contains only a single unique patient the procedure routes
that entire class to the training set and emits a warning. In the current
corpus this situation affects `dry_eye`, where eight training images and
five held-out images come from two different patients respectively, so no
fallback is triggered; the branch exists for robustness in case future
ingestion reduces a class to one patient.

## Resulting split

The table below summarises the split actually produced by
`tearcls/data_split.py` on the current raw directory (seed 3407).
"Patients" counts unique patient codes, "Images" counts BMPs.

| Class                 | Train images | Train patients | Held-out images | Held-out patients |
|-----------------------|-------------:|---------------:|----------------:|------------------:|
| `diabetes`            | 19           | 3              | 8               | 1                 |
| `glaucoma`            | 23           | 4              | 12              | 1                 |
| `multiple_sclerosis`  | 78           | 10             | 17              | 2                 |
| `dry_eye`             | 8            | 1              | 5               | 1                 |
| `healthy`             | 61           | 15             | 9               | 4                 |
| **Total**             | **189**      | **33**         | **51**          | **9**             |

Training therefore covers 189 images from 33 patients and the held-out
evaluation set covers 51 images from 9 patients, with every class
represented in both partitions. The overall training fraction at the
image level is 78.8%, close to the targeted 80% although not identical,
because the ratio is enforced on patient counts and not on image counts
and some patients contribute more scans than others.

## Leakage audit

The splitting script verifies two invariants at the end of every run and
aborts if either is violated:

1. `train ∩ val = 0` and `train ∩ test = 0` at the patient-code level.
2. Every held-out row appears exactly twice in the output CSV, once per
   split label, with identical metadata otherwise.

For the split used in this paper both invariants hold. All subsequent
training, hyperparameter selection and reported test metrics are produced
against this single fixed split; cross-validation is not performed,
because the small number of unique patients per class would lead to
folds dominated by stochastic patient assignment rather than by genuine
model differences.

## Reproducibility

The split is fully determined by the content of `data_raw/` together with
the three constants at the top of `tearcls/data_split.py` (`SEED`,
`EVAL_FRAC`, `CLASS_MAP`). Regenerating it from a fresh clone requires a
single invocation of that script and produces a byte-identical
`data/splits.csv`. No split decision is made inside the training loops
themselves; every experiment consumes the CSV unchanged.

---

## TODO: analyses to add before camera-ready

- [ ] Bar chart of images per class broken down by split, as a visual
      companion to the table above.
- [ ] Distribution of scans per patient in the training and held-out
      partitions, to show that neither partition is dominated by a single
      prolific patient.
- [ ] Sensitivity analysis: re-run the split with several seeds and
      report mean ± std of downstream test accuracy, to quantify how much
      the single-seed number depends on which 9 patients happen to be
      held out.
- [ ] Justification paragraph for not using k-fold cross-validation,
      backed by a small experiment showing fold-to-fold variance at the
      available corpus size.
- [ ] Discussion of the decision to merge val and test, including the
      risk that hyperparameters tuned on val are implicitly tuned on
      test, and a proposal for an external held-out cohort to be
      collected for the journal version.
- [ ] Check for acquisition-date overlap between train and held-out
      patients per class, to confirm there is no temporal signal the
      model can exploit as a proxy for diagnosis.
