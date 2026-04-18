# Data

## Source and modality

The dataset consists of Atomic Force Microscopy (AFM) scans of human tear
films, acquired on a Bruker Dimension Icon NanoScope (scanner head type SG,
serial `1b814`). Unlike light-microscopy or tear-ferning imagery used in some
prior tear-film studies, the underlying signal here is a nanometre-scale
height map of the dried tear residue surface, rendered as a false-colour
bitmap by the vendor acquisition software. Each scan was collected by a
clinician at a partner ophthalmology site; the raw material was provided as
an archive of per-patient folders, one folder hierarchy per diagnostic class.

## Classes

The task is framed as single-label classification over five clinically
defined groups. The folder names in the raw archive are in Slovak and are
normalised to English identifiers used throughout the code and the rest of
this paper:

| Raw folder            | Class label           | Samples (BMPs) |
|-----------------------|-----------------------|----------------|
| `ZdraviLudia`         | `healthy`             | 70             |
| `SklerózaMultiplex`   | `multiple_sclerosis`  | 95             |
| `PGOV_Glaukom`        | `glaucoma`            | 35             |
| `Diabetes`            | `diabetes`            | 27             |
| `SucheOko`            | `dry_eye`             | 13             |
| **Total**             |                       | **240**        |

The class distribution is markedly imbalanced: the largest class
(`multiple_sclerosis`) contains roughly seven times as many samples as the
smallest (`dry_eye`), and the three smallest classes together account for
only about 31% of the corpus. This imbalance reflects the prevalence of
each pathology in the referring clinic's patient population rather than any
sampling design choice, and it is preserved unchanged in all downstream
experiments.

## Per-sample artefacts

Each scan is represented on disk by two sibling files sharing a common
stem:

1. A 24-bit Windows BMP rendered by the NanoScope software. Native
   resolutions are close to 704×575 pixels, with a false-colour lookup table
   mapping measured height to RGB. The BMP is the only visual modality
   consumed by the image encoders.
2. A binary AFM container (no extension, or numeric extension such as
   `.010`) holding an ASCII metadata header of roughly 34 KB followed by
   the raw 16-bit height samples. The header encodes acquisition settings
   (scan size, rate, line count, gains, Z-sensitivity, tip and cantilever
   identifiers, session timestamp, and per-channel calibration constants),
   which we treat as a structured text modality paired with the image.

Patient identity is not stored inside the files; it is recovered from the
filename stem (e.g. `DM_01.03.2024_LO`, `37_DM`), which encodes an
anonymised patient code together with an acquisition date and, in most
cases, an eye indicator (`LO`/`PO`). Several patients contributed multiple
scans in a single session, so the number of unique patients per class is
substantially smaller than the number of images, a fact that is material
for the splitting strategy (discussed separately).

## Cohort characteristics

The dataset is small by modern computer-vision standards — 240 scans in
total — and was not collected under a predefined imaging protocol targeting
machine learning. Scan regions, magnifications and line counts vary between
sessions, and a fraction of the scans contain instrument artefacts typical
of contact-mode AFM (scan-line streaks, tip-contamination bands, local
piezo drift). All such artefacts are retained in the published corpus; no
sample was excluded on quality grounds.

## Preprocessing applied before modelling

The only transformation applied before any model sees a sample is a
lossless conversion of the BMP to PNG at native resolution, together with
the extraction of a curated subset of AFM header fields into a plain-text
digest. No resampling, normalisation, colour-space change, or content
editing is performed at this stage; resizing and tensor normalisation are
deferred to the individual model's preprocessing pipeline.

---

## TODO: exploratory data analysis to add before camera-ready

The following analyses would strengthen this section and are currently
missing:

- [ ] Per-class image-count bar chart (visual counterpart to the table
      above).
- [ ] Unique-patient counts per class, and distribution of scans per
      patient, to justify the patient-stratified split quantitatively.
- [ ] Acquisition-date histogram per class, to expose any temporal
      confounding between diagnosis and imaging session.
- [ ] Distribution of native BMP resolutions and aspect ratios.
- [ ] Intensity / per-channel colour statistics per class (mean, std,
      histograms) to check for trivially separable colour-LUT shortcuts.
- [ ] Summary statistics for the AFM header fields actually consumed by
      the model (scan size, scan rate, line count, Z-sensitivity): mean,
      std, and overlap across classes.
- [ ] A small qualitative figure — one representative scan per class —
      with a short caption describing the dominant visual features a
      clinician would point to.
- [ ] Inter-patient vs. intra-patient similarity analysis (e.g. pHash or
      embedding distance), to quantify the leakage risk that motivated the
      patient-stratified split.
- [ ] Note any known acquisition-protocol differences between classes
      (operator, calendar window, scanner tip batch) if such metadata can
      be recovered from the clinical side.
