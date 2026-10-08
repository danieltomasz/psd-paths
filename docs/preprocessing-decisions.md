# Preprocessing decisions for v0.3: filtering, line noise and ICA

This document records why the preprocessing changed between v0.2.3 and v0.3,
what was tested, and the literature behind each choice. It covers the filter
band, the line-noise notch, the ICA (algorithm, number of components, ICLabel
thresholds), eye artifacts, and how specparam fit quality is reported.

All tests used the 38 PATHS subjects in `data/bids`, with `runs/v0.2.3` as the
reference. Unless stated otherwise, numbers are medians over subjects. The
parameter values themselves are in `settings.toml` and in the parameter cells
of the notebooks in `templates/`; the changes are listed in `CHANGELOG.md`.

## Summary

| | v0.2.3 | v0.3 | Section |
| --- | --- | --- | --- |
| Band-pass (step 01) | 1-40 Hz | 1-100 Hz | [1](#1-filter-band-1-100-hz-instead-of-1-40-hz) |
| Notch at 50/100 Hz | 0.25 Hz wide (mne default) | 1 Hz wide | [2](#2-line-noise-notch-width) |
| ICA components | as many as explain 99% of the variance, at most 40 (9-40 per subject) | 44 for every subject | [3.2](#32-number-of-components) |
| ICA algorithm | Picard, extended, ortho=False | unchanged | [3.1](#31-algorithm) |
| ICLabel threshold, muscle | 0.80 | 0.70 | [3.3](#33-iclabel-thresholds) |
| ICLabel threshold, eye blink | 0.50 | unchanged | [3.3](#33-iclabel-thresholds) |
| Eye artifacts | removed with ICA | unchanged (no segment rejection) | [4](#4-eye-artifacts-correction-not-rejection) |
| Autoreject before ICA (step 02) | fitted at 125 Hz | unchanged | [1.3](#13-effect-on-bad-channels-and-autoreject) |
| Component review widget (step 03) | always executed | only with `review = True` | [3.5](#35-manual-review) |
| Group report | R² rule | R² rule plus fit error columns (shown, not used) | [5](#5-fit-quality-r²-and-fit-error) |

## Background: what was wrong with v0.2.3

- **ICLabel got data it was not trained on.** ICLabel computes its features
  (component spectra) on 1-100 Hz data (Pion-Tonachini et al. 2019, p. 4), and
  mne-icalabel warns when the data is filtered otherwise. v0.2.3 filtered
  1-40 Hz, so ICLabel could not see the muscle band.
- **ICA sometimes added high-frequency power.** In 12 of 38 subjects, power in
  30-40 Hz was higher after ICA than before. In a test on 17 subjects, every
  subject had channels where beta or 30-40 Hz power rose by more than 10%
  with the v0.2.3 setup (1-40 Hz, variance-based number of components), and
  12 of 17 with a 1-100 Hz ICA. Median change in 30-40 Hz: +10.5% (v0.2.3
  setup) vs -6.3% (1-100 Hz, 40 components).
- **The number of components depended on blinks.** "As many components as
  explain 99% of the variance" gave 17 of 38 subjects fewer than 40, as few as
  9 (sub-137), because blink components carry most of the variance in some
  recordings.
- **Muscle activity stayed in the data.** v0.2.3 removed a median of one
  muscle component per subject. sub-106 and sub-129 were excluded by the R²
  rule; sub-106 recovered when its muscle components were removed.

## 1. Filter band: 1-100 Hz instead of 1-40 Hz

### 1.1 Why the low-pass was 40 Hz

The 40 Hz low-pass was there because the 50 Hz notch did not remove the line
noise. Section 2 shows why (the default notch is too narrow) and fixes it,
which removes the reason for the 40 Hz low-pass.

### 1.2 Why 1-100 Hz

- ICLabel is trained on average-referenced data filtered 1-100 Hz
  (Pion-Tonachini et al. 2019, p. 4: power spectral densities from 1 to 100 Hz,
  common average reference).
- MNE-BIDS-Pipeline (1.10.1, installed here) requires `ica_l_freq = 1` and
  `ica_h_freq = 100` when ICLabel is used, on top of the main filter, so the
  main low-pass must be at least 100 Hz.
- Other pipelines detect bad channels and segments on broadband data: PREP
  (Bigdely-Shamlo et al. 2015; its noise criterion compares power above and
  below 50 Hz and was tuned on data with only a 1 Hz high-pass), RELAX (Bailey
  et al. 2023, p. 4: 1-80 Hz band-pass and a 47-53 Hz notch), Automagic
  (Pedroni et al. 2019; Tröndle & Langer 2026, p. 27: 0.5 Hz high-pass and
  line-noise removal).
- Muscle activity spans about 20-300 Hz (Muthukumaraswamy 2013). With a 40 Hz
  low-pass, ICA and ICLabel see only its lower end.
- The high-pass stays at 1 Hz: for stationary recordings with many channels,
  Klug & Gramann (2021, p. 13) recommend up to 1.25 Hz.
- specparam fits 2-35 Hz, so the spectra it uses are not affected by what
  happens above 40 Hz. Nothing needs to be filtered after ICA.

### 1.3 Effect on bad channels and autoreject

Tested with a run that changed only the filter (1-100 Hz, 1 Hz notch, ICA as
in v0.2.3), compared with `runs/v0.2.3`:

| | v0.2.3 (1-40 Hz) | 1-100 Hz | Subjects more / fewer / same |
| --- | --- | --- | --- |
| Bad channels (step 01) | 9.5 | 9 | 8 / 12 / 18 |
| Peak-to-peak amplitude of epochs | 43 µV | 51 µV | 38 / 0 / 0 |
| Epochs rejected before ICA (step 02) | 6.5 | 6.5 | 6 / 15 / 17 |
| Epochs rejected after ICA (step 04) | 0 | 0 | 6 / 4 / 28 |
| Final epochs | 78 | 77.5 | 17 / 8 / 13 |

- Autoreject learns its thresholds per channel from each recording, so higher
  amplitudes raise the thresholds too. The change in rejected epochs was not
  related to how much muscle activity a subject had (Spearman rho 0.13 in
  step 02, -0.10 in step 04).
- The largest single changes: sub-108 13 -> 8 bad channels; sub-140 18 -> 23
  epochs rejected in step 02; sub-114 1 -> 6 in step 04.
- `lof_threshold` (4.0) and `fallback_ptp_uv` were set on 1-40 Hz data. Bad
  channel counts barely changed, so they were kept.

**Autoreject in step 02 on 1-100 Hz?** Step 02 fits autoreject on a copy
resampled to 125 Hz, so it sees up to about 60 Hz. Fitted at 250 Hz instead
(sees 1-100 Hz), it rejected 280 epochs in total instead of 299; 14 epochs were
rejected only at 250 Hz (7 of them with clearly raised muscle power), 33 only
at 125 Hz. The 125 Hz version reproduced the pipeline in 38/38 subjects.
Step 02 removes gross artifacts before ICA, ICA needs examples of muscle
activity to separate it, and step 04 already rejects on 1-100 Hz data after
ICA. Decision: keep 125 Hz.

**Is the muscle activity bursty or constant?** Power in 55-95 Hz per 5 s epoch
(1-100 Hz data, kept epochs): between subjects it differs about 55-fold; within
a subject it varies by about ±12% (median absolute deviation of log power
about 0.05). In sub-129, the loudest epoch is 1.5 times its median. The muscle
activity is tonic, so rejecting segments cannot remove it; ICA can. Klug, Berg
& Gramann (2023) found that moderate cleaning before ICA gives the best
decompositions.

## 2. Line noise: notch width

### 2.1 The default notch is too narrow

mne's FIR notch is `freq / 200` wide by default: 0.25 Hz at 50 Hz. Measured as
the power at 50.0 Hz (±0.25 Hz) above 45-47 and 53-55 Hz, per good channel,
before re-referencing (1-100 Hz data):

| Notch width | Subjects with any channel > 6 dB | Worst subject: median channel | Worst subject: 95th percentile channel |
| --- | --- | --- | --- |
| none | 38 | +50.5 dB | +53.3 dB |
| 0.25 Hz (default) | 14 | +8.7 dB | +9.4 dB |
| 0.5 Hz | 0 | +1.8 dB | +3.1 dB |
| 1 Hz | 0 | -1.9 dB | +1.2 dB |
| 1.5 Hz | 0 | -7.6 dB | -5.9 dB |
| 2 Hz | 0 | -7.5 dB | -4.4 dB |

With the default notch, the worst subjects were sub-108, 140, 139, 107 and 101
(34 to 185 channels above 6 dB). MNE's `spectrum_fit` method (the CleanLine
approach used in PREP) left 50 Hz above 6 dB in all 38 subjects (worst subject,
median channel +28.8 dB): it assumes a stable sinusoid, which the line noise
here is not. Earlier, on 4 subjects, ZapLine implementations were tested
(de Cheveigné 2020; Klug & Kloosterman 2022). In sub-101 (+39 dB without
cleaning), the residual 50 Hz peak was +35.7 dB with meegkit's ZapLine
(one component), +0.7 dB with mne-denoise's ZapLine, +4.6 dB with its adaptive
mode, and +36.6 dB with pyzaplineplus (which did not settle on a solution);
the default FIR notch gave -0.4 dB. The FIR notch was kept.

### 2.2 The line noise has a skirt and sidebands

- **Skirt.** Without a notch, 46-54 Hz is 4-15 dB above 43-45/55-57 Hz in the
  worst subjects (sub-140, sub-108; median over channels). A 1 Hz notch removes
  the centre, a 4 Hz notch leaves +9 dB at 47.5 and 52.5 Hz, a 6 Hz notch
  (47-53 Hz) leaves +4-6 dB at its outer edges.
- **In power, the skirt is small.** Of the excess power in 45-55 Hz (above
  40-43/57-60 Hz), a 1 Hz notch removes 99.7% in the median channel. In the
  worst subject 38% is left with 1 Hz, 33% with 4 Hz and 26% with 6 Hz.
- **Sidebands from switching line noise.** Example: sub-101, component 7
  (labelled muscle by ICLabel, p = 0.94). 41% of its power is in 49-51 Hz;
  at 0.25 Hz resolution, the 1 Hz notch removes 49.5-50.5 Hz (-14 to -22 dB),
  but there are peaks at 49.0 and 51.0 Hz. The 48-52 Hz and 60-90 Hz envelopes
  of its main channel move together (r = 0.85 over 1 s windows; 48-52 Hz up to
  29 times its median). The component sits on a small electrode cluster (E109,
  E108, E99, E116, E107) next to E100 and E110, which were already marked bad.
  This is an electrode losing and regaining contact, picking up line noise and
  broadband noise in bursts; it is not muscle. Removing it is right, the label
  is not.

### 2.3 A wider notch breaks ICLabel's muscle detection

A wider notch leaves a gap in the spectrum of every ICA component. ICLabel reads
component spectra from 1 to 100 Hz and was not trained on such gaps. Test: one
ICA (44 components) fitted on 1 Hz-notch data per subject, then ICLabel run on
the same components with the data notched 1, 2 or 6 Hz wide:

| ICLabel run on data with notch | Muscle components with p > 0.7 (all subjects) | Components with "other" as top class | Components removed |
| --- | --- | --- | --- |
| 1 Hz | 350 | 240 | 587 |
| 2 Hz | 261 | 208 | 494 |
| 6 Hz | 71 | 575 | 311 |
| full pipeline with a 6 Hz notch (own ICA) | 55 | 584 | 342 |

The same components are called muscle with a 1 Hz notch and "other" with a
6 Hz notch. A full pipeline run with a 6 Hz notch removed a median of 1 muscle
component per subject (as in v0.2.3), its exponent changes no longer followed
the subjects' muscle level (Spearman rho 0.12, against 0.81 with a 1 Hz notch),
and sub-129 was excluded again. RELAX also uses a 47-53 Hz notch before ICLabel
(Bailey et al. 2023), but cleans muscle with a multichannel Wiener filter
before ICA, so it depends less on ICLabel's muscle class.

### 2.4 Choice of width

1, 1.5 and 2 Hz were compared on the 12 most problematic subjects: the worst
line noise (sub-108, 140, 107, 139, 101) and the most muscle activity or poorest
fits (sub-106, 129, 124, 145, 113, 109, 170). Each width went through its own
ICA as in the pipeline (44 components, ICLabel thresholds from `settings.toml`),
then interpolation and specparam 2-35 Hz.

| | 1 Hz | 1.5 Hz | 2 Hz |
| --- | --- | --- | --- |
| Line-noise power left in 45-55 Hz (median channel) | 0.35% | 0.32% | 0.29% |
| Channels with a residual peak > 3 dB in 47.5-52.5 Hz (total) | 1221 | 1166 | 1097 |
| Muscle components with p > 0.7 (total) | 130 | 113 | 94 |
| Components removed (total) | 215 | 200 | 182 |
| Fit error (median) | 0.0344 | 0.0342 | 0.0334 |
| Exponent, change per subject against 1 Hz (median) | | -0.01 | -0.04 |

- A wider notch removes almost no more line noise. The skirt is wider than
  2 Hz: sub-107, 108, 139 and 140 keep residual peaks in 180-193 channels with
  every width.
- ICLabel finds fewer muscle components: 13% fewer at 1.5 Hz, 28% fewer at
  2 Hz (on all 38 subjects, with ICLabel run on the same components: 25% fewer
  at 2 Hz, section 2.3).
- With fewer muscle components removed, the exponent drops in the subjects with
  the most muscle activity (sub-145: 0.95, 0.85, 0.78; sub-124: 1.38, 1.15,
  1.14), and sub-129's fits get worse at 2 Hz (good-fit share 0.80 -> 0.61).
  The median fit error does not change.

**Decision: 1 Hz.** It removes the line-noise peak (99.7% of the excess power in
the median channel) and keeps ICLabel's muscle detection. The remaining skirt
is weak in power, lies outside the specparam range (2-35 Hz) and outside the
muscle band used for quality checks (55-95 Hz). The order of tests was: 1 Hz
first, then 6 Hz to remove the skirt (as in RELAX), then back to 1 Hz after the
ICLabel effect was found.

## 3. ICA

### 3.1 Algorithm

ICLabel was trained on extended Infomax decompositions. mne-icalabel accepts
two equivalents and warns for anything else (`mne_icalabel/iclabel/features.py`,
lines 73-85): `method="infomax"` with `extended=True`, or `method="picard"` with
`ortho=False, extended=True`. The pipeline uses the second
(`compute_ica_limited` in eeg-spectral), with Infomax as fallback. Picard
optimises the same objective as Infomax, faster (Ablin et al. 2018).

### 3.2 Number of components

**Not by variance.** Reducing the data with PCA before ICA, even by 1% of the
variance, lowers the quality and stability of the decomposition (Artoni et al.
2018). The variance rule in v0.2.3 was the worst case: subjects dominated by
blinks got 9-15 components.

**Limited by data length.** ICA needs enough samples per parameter. HAPPE
(Gabard-Durnam et al. 2018) uses at least 30 × (number of components)² samples,
after Onton & Makeig, and states that 128- and 256-channel recordings of usual
length do not have enough data for ICA on all channels; its example reduces to
40 of 128 channels for a resting recording at 250 Hz. MNE-BIDS-Pipeline uses
all components by default (`ica_n_components = None`), which ignores data
length.

The PATHS epochs overlap by 1.5 s, so overlapping parts were counted once:
69,875 unique samples per subject (median; minimum 59,750 in sub-132, maximum
141,375 in sub-114).

| Components | Samples needed (30 × n²) | Subjects that meet it |
| --- | --- | --- |
| 40 | 48,000 | 38/38 |
| 44 | 58,080 | 38/38 |
| 45 | 60,750 | 35/38 (sub-132, 140, 135 at 29.5-29.9) |
| 48 | 69,120 | 21/38 |
| 50 | 75,000 | 6/38 |

**Tested with 40, 50 and 60 components** (1-100 Hz data, 1 Hz notch; each fitted
twice, seeds 97 and 98; muscle threshold 0.5 for the cleaning):

| | 40 | 50 | 60 |
| --- | --- | --- | --- |
| Unique samples per squared component (min) | 44 (37) | 28 (24) | 19 (17) |
| Components found again with the other seed, \|r\| > 0.95 (mean share) | 0.958 | 0.946 | 0.930 |
| Subjects with less than 90% found again | 7 | 8 | 11 |
| Brain components (ICLabel p > 0.5) | 11 | 12 | 13 |
| Brain components found again | 100% | 100% | 100% |
| Muscle components removed | 8 | 11.5 | 13.5 |
| 30-40 Hz power after ICA | -16% | -28% | -26% |
| Subjects with good-fit share < 0.9 | 3 | 2 | 1 |

All decompositions converged. The specparam fit error was the same for 40 and
50 components. **Decision: 44**, the largest number that meets the 30 × n² rule
in every subject; 50 would remove more of the muscle band, but exceeds the rule
in 32 of 38 subjects and is slower to review. Step 03 now prints the unique
samples per squared component and adds a warning to the report below 30
(`runs/v0.3.0-dev`: minimum 30.9, median 36.0).

### 3.3 ICLabel thresholds

A component is removed if its top ICLabel class is an artifact class and its
probability is above that class's threshold.

**Eye blink: 0.50 (unchanged).** With 0.80, blink components stayed in for 6 of
20 subjects; sub-109 kept two eye components (p = 0.74 and 0.54) carrying 58%
of its variance.

**Muscle: 0.70 (was 0.80).**

- Reference points: MNE-BIDS-Pipeline uses 0.8 for every class. The ICLabel
  paper's thresholds that maximise detection (Pion-Tonachini et al. 2019,
  Table 3) are much lower: muscle 0.18-0.30, eye 0.04-0.13.
- Every component ICLabel called muscle was checked against two criteria
  ICLabel does not use: the spectral slope 7-70 Hz (muscle has a flat or rising
  spectrum; RELAX uses slope > -0.59 after Fitzgibbon et al., here applied to
  component spectra) and an alpha peak (8-12 Hz at least 0.2 log10 above 5-7
  and 14-16 Hz, a sign of mixed-in brain activity). 40 components, 38 subjects:

| Muscle probability | Components | Muscle-like slope | With alpha peak | Brain probability > 0.3 |
| --- | --- | --- | --- | --- |
| 0.3-0.5 | 66 | 68% | 3 | 12 |
| 0.5-0.6 | 35 | 91% | 1 | 9 |
| 0.6-0.7 | 39 | 90% | 1 | 7 |
| 0.7-0.8 | 41 | 93% | 2 | 0 |
| 0.8-0.9 | 63 | 98% | 2 | 0 |
| 0.9-1.0 | 181 | 100% | 11 | 0 |

- Components between 0.7 and 0.8 look like muscle and none is ambiguous
  (brain p > 0.3 is impossible above 0.7, since the probabilities sum to 1).
  Between 0.5 and 0.7, 16 components have brain p > 0.3.
- **Components with an alpha peak are mostly above 0.8** (13 of 17), so no
  threshold separates them; only a review can. They carry 1-7% of the variance
  (sub-129, 132 and 120 the largest).
- **specparam fit error by muscle threshold** (50 components, all good
  channels of all subjects):

| Threshold | Muscle components removed | Fit error (MAE), median | Fit error, worst 10% | R², worst 10% | Channels with R² ≥ 0.9 | Exponent |
| --- | --- | --- | --- | --- | --- | --- |
| no muscle removal | 0 | 0.0351 | 0.0587 | 0.916 | 91.8% | 1.10 |
| 0.9 | 5.5 | 0.0343 | 0.0539 | 0.948 | 95.0% | 1.16 |
| 0.8 | 7 | 0.0337 | 0.0523 | 0.957 | 96.5% | 1.17 |
| 0.7 | 8.5 | 0.0338 | 0.0522 | 0.961 | 97.2% | 1.19 |
| 0.5 | 11.5 | 0.0335 | 0.0523 | 0.964 | 97.7% | 1.22 |

  Removing muscle components improves the fits; below 0.8 the error is flat
  (0.7 vs 0.8: all 38 subjects within 5%). R² keeps rising, but mostly because
  the exponent gets steeper (section 5). Between 0.5 and 0.7 the remaining
  difference is about 0.03 in the exponent, against removing 16 ambiguous
  components without review. Decision: 0.70.

### 3.4 Limits of ICLabel for this data

- ICLabel's training data contained few older adults and no major brain
  pathology (Pion-Tonachini et al. 2019, section 5.4). PATHS participants are
  older adults after stroke.
- Example of a wrong label: sub-101 component 7 (section 2.2), called muscle
  (p = 0.94) but a bad-contact electrode cluster.
- The thresholds and component numbers here should be checked against the
  manually labelled pilot dataset (gold standard) that Sara Zago proposed.
  Chaumon et al. (2015) give a practical guide for manual selection.

### 3.5 Manual review

The component review widget at the end of step 03 now runs only with
`review = True` in the notebook's parameter cell. In a pipeline run it is
skipped: a widget cannot be used there, and in the run with a 6 Hz notch it
left sub-105 hanging in that cell for 24 minutes. The review spectrum goes to
100 Hz (`psd_fmax=100.0`; the eeg-spectral default is 45 Hz).

## 4. Eye artifacts: correction, not rejection

**How often.** Eye events (peaks above 5 robust SD in the strongest eye
component of the v0.2.3 ICA): median 15 per minute (2-51; sub-170 51). 64% of
the 5 s epochs contain one (10-100%); 22 of 38 subjects have one in more than
half of their epochs. Rejecting those epochs would remove about 60% of the data,
leave 12 subjects with fewer than 15 epochs (none in sub-170 and sub-112), and
make the amount of data depend on each subject's blink rate.

**Does removing the eye components change alpha or the exponent?** In the 24
subjects with at least 15 blink-free epochs, the same blink-free epochs were
analysed with the eye components kept and removed (v0.2.3 data):

| | Eye components kept | Removed | Change per subject (median) |
| --- | --- | --- | --- |
| Alpha peak power | 0.439 | 0.442 | +0.004 (Spearman rho 0.97) |
| Alpha peak frequency | 8.98 Hz | 8.84 Hz | -0.001 Hz (rho 0.94) |
| Exponent, front quarter of channels | 1.24 | 0.88 | -0.13 (lower in 22/24) |
| Exponent, back quarter of channels | 1.17 | 0.98 | -0.07 (lower in 23/24) |
| Power 2-4 Hz | | | -11% |

Alpha is not affected. The exponent is lower even in blink-free epochs, twice
as much at the front as at the back: slow eye movements and drift continue
between blinks and raise low-frequency power. Removing them is the intended
correction; eye activity makes the aperiodic exponent steeper (Tröndle & Langer
2026). Rejecting blink segments would not remove this slower eye activity. Some
brain delta in the eye components cannot be ruled out, but the front-to-back
gradient points to the eyes.

**Other pipelines** correct eye artifacts rather than rejecting segments:
RELAX (multichannel Wiener filter, then wavelet-enhanced ICA on ICLabel artifact
components; Bailey et al. 2023), HAPPE (wavelet-enhanced ICA and ICA;
Gabard-Durnam et al. 2018), Automagic (ICA-based; Pedroni et al. 2019). Artoni
& Michel (2025) found removing ocular components essential, and more aggressive
cleaning added little. Issa & Juhasz (2019) found that removing whole eye
components can distort the spectrum and that correcting only during eye
activity (wavelet-based) did better; Cassani et al. (2014) found wavelet-enhanced
ICA better than segment rejection in resting EEG of older adults with
Alzheimer's disease. Dimigen (2020) showed that standard ICA can leave eye
artifacts and distort neural activity in free-viewing tasks, and how training
the ICA differently fixes it. Delorme (2023) found that automated ICA removal of
eye and muscle components did not reliably improve ERP results.

Decision: keep removing eye components with ICA; no blink segment rejection.
Wavelet-based correction of the eye components (as in RELAX) is the next step
if needed. Blink rate per subject can be used as a covariate (see section 7).

## 5. Fit quality: R² and fit error

Across channels, specparam's R² correlates with the exponent (Spearman
rho 0.63) and its fit error does not (rho -0.11): a steeper spectrum leaves more
variance to explain, so R² rises even when the fit is no closer. The group rule
("Exclude" if fewer than 50% of channels have R² ≥ 0.9) therefore partly
excludes subjects with flat spectra rather than poor fits. Example: in the
filter-only test run, sub-109 had the largest fit error (0.054, nearly twice the
median) and passed the R² rule with all channels.

The group report now also lists `specparam_error_median` (median fit error over
measured channels; interpolated channels left out) and `specparam_error_z`
(robust z-score against the other subjects of the run). They are shown, not
used: whether to add an error-based rule is still open.

## 6. Results: v0.3 compared with v0.2.3

`runs/v0.3.0-dev` (1-100 Hz, 1 Hz notch, 44 components, muscle threshold 0.70;
eeg-spectral 0.2.4) against `runs/v0.2.3`. 38 subjects, 190/190 steps. Medians
over subjects, measured channels only (interpolated channels left out).

| | v0.2.3 | v0.3.0-dev |
| --- | --- | --- |
| ICA components | 40 (17 subjects fewer, as few as 9) | 44 for everyone |
| Unique samples per squared component | | 36 (minimum 30.9) |
| Components removed: total / muscle / eye | 6 / 1 / 2 | 14.5 / 8 / 3 |
| Final epochs | 78 | 76.5 |
| Exponent | 1.03 | 1.16 |
| Offset | -9.09 | -9.03 |
| Exponent, edge minus top quarter of channels | -0.19 | -0.15 |
| Fit error (MAE) | 0.033 | 0.032 |
| Subjects with good-fit share < 0.9 | 6 | 4 |
| Subjects marked Exclude | sub-106, sub-129 | none |
| Alpha peak frequency | 8.87 Hz | 9.00 Hz |
| Alpha peak power above the aperiodic fit | 0.44 | 0.49 |
| Exponent vs age (Spearman, n = 38) | -0.38 | -0.44 |

- **The order of subjects is kept.** Between runs, Spearman rho is 0.88 for the
  exponent, 0.93 for the offset, 0.99 for the alpha peak frequency.
- **The exponent rises with each subject's muscle activity** (rho 0.80 between
  the change and the subject's 55-95 Hz power): sub-124 0.71 -> 1.35, sub-106
  0.54 -> 0.94, sub-129 0.46 -> 0.71, sub-168 0.83 -> 1.15 (11 components in
  v0.2.3). Median change +0.09, higher in 31 of 38 subjects. This is the
  direction expected when muscle activity is removed (Tröndle & Langer 2026).
- **It falls in 7 subjects**, most in sub-109 (1.76 -> 1.19) and sub-137
  (1.80 -> 1.70). Both had few components in v0.2.3 (15 and 9); with 44, more
  eye and channel-noise components are separated and removed. sub-109 had the
  largest fit error in v0.2.3 (0.054), now 0.040. The size of this change makes
  sub-109 a candidate for manual review.
- **Fits:** fit error lower in 26 subjects; sub-129 good-fit share 0.46 -> 0.85,
  sub-106 0.47 -> 0.65; no subject below the exclusion limit. Largest fit errors
  now: sub-127 and sub-112 (0.045, error z-score 2.1).
- **Alpha is kept:** peak frequency unchanged per subject, peak power slightly
  higher (it is measured above the aperiodic fit, which is steeper after
  cleaning).
- **Age:** the known decrease of the exponent with age is slightly stronger
  (rho -0.38 -> -0.44).

Test runs made on the way (deleted; their numbers are quoted in the sections
above): 1-100 Hz with the 1 Hz notch and the v0.2.3 ICA (section 1.3); 50
components with muscle threshold 0.70 (exponent 1.17, no exclusions, very
similar to the final run); 44 components with a 6 Hz notch (section 2.3).

## 7. Open questions

- Validate the ICA settings against the manually labelled pilot dataset.
- Flag muscle components with an alpha peak for manual review (not yet
  implemented).
- Covariates for the group analysis: muscle components removed and their
  variance, blink rate (computed for all 38 subjects), following Tröndle &
  Langer (2026), who correct aperiodic estimates for artifact load by
  regression.
- Whether to use the fit error in the exclusion rule.
- Wavelet-enhanced correction of artifact components (RELAX) instead of
  removing whole components.
- Line-noise sidebands from bad electrode contact (section 2.2) end up in ICA
  components that ICLabel may misclassify.

## References

- Ablin P, Cardoso J-F, Gramfort A (2018). Faster independent component
  analysis by preconditioning with Hessian approximations. *IEEE Transactions on
  Signal Processing* 66(15):4040-4049. <https://doi.org/10.1109/TSP.2018.2844203>
- Artoni F, Delorme A, Makeig S (2018). Applying dimension reduction to EEG data
  by principal component analysis reduces the quality of its subsequent
  independent component decomposition. *NeuroImage* 175:176-187.
  <https://doi.org/10.1016/j.neuroimage.2018.03.016>
- Artoni F, Michel CM (2025). How does independent component analysis
  preprocessing affect EEG microstates? *Brain Topography* 38:26.
  <https://doi.org/10.1007/s10548-024-01098-4>
- Bailey NW, Biabani M, Hill AT, et al. (2023). Introducing RELAX: an automated
  pre-processing pipeline for cleaning EEG data. Part 1: algorithm and
  application to oscillations. *Clinical Neurophysiology* 149:178-201.
  <https://doi.org/10.1016/j.clinph.2023.01.017>
- Bigdely-Shamlo N, Mullen T, Kothe C, Su K-M, Robbins KA (2015). The PREP
  pipeline: standardized preprocessing for large-scale EEG analysis. *Frontiers
  in Neuroinformatics* 9:16. <https://doi.org/10.3389/fninf.2015.00016>
- Cassani R, Falk TH, Fraga FJ, Kanda PAM, Anghinah R (2014). The effects of
  automated artifact removal algorithms on electroencephalography-based
  Alzheimer's disease diagnosis. *Frontiers in Aging Neuroscience* 6:55.
  <https://doi.org/10.3389/fnagi.2014.00055>
- Chaumon M, Bishop DVM, Busch NA (2015). A practical guide to the selection of
  independent components of the electroencephalogram for artifact correction.
  *Journal of Neuroscience Methods* 250:47-63.
  <https://doi.org/10.1016/j.jneumeth.2015.02.025>
- de Cheveigné A (2020). ZapLine: a simple and effective method to remove power
  line artifacts. *NeuroImage* 207:116356.
  <https://doi.org/10.1016/j.neuroimage.2019.116356>
- Delorme A (2023). EEG is better left alone. *Scientific Reports* 13:2372.
  <https://doi.org/10.1038/s41598-023-27528-0>
- Dimigen O (2020). Optimizing the ICA-based removal of ocular EEG artifacts
  from free viewing experiments. *NeuroImage* 207:116117.
  <https://doi.org/10.1016/j.neuroimage.2019.116117>
- Fitzgibbon SP et al. (2016): muscle criterion (log-log spectral slope), as
  used in Bailey et al. (2023); not read directly.
- Gabard-Durnam LJ, Mendez Leal AS, Wilkinson CL, Levin AR (2018). The Harvard
  Automated Processing Pipeline for Electroencephalography (HAPPE). *Frontiers in
  Neuroscience* 12:97. <https://doi.org/10.3389/fnins.2018.00097>
- Issa MF, Juhasz Z (2019). Improved EOG artifact removal using wavelet enhanced
  independent component analysis. *Brain Sciences* 9(12):355.
  <https://doi.org/10.3390/brainsci9120355>
- Klug M, Gramann K (2021). Identifying key factors for improving ICA-based
  decomposition of EEG data in mobile and stationary experiments. *European
  Journal of Neuroscience* 54(12):8406-8420. <https://doi.org/10.1111/ejn.14992>
- Klug M, Berg T, Gramann K (2023). No need for extensive artifact rejection for
  ICA: a multi-study evaluation on stationary and mobile EEG datasets. bioRxiv
  preprint. <https://doi.org/10.1101/2022.09.13.507772>
- Klug M, Kloosterman NA (2022). Zapline-plus: a Zapline extension for automatic
  and adaptive removal of frequency-specific noise artifacts in M/EEG. *Human
  Brain Mapping* 43(9):2743-2758. <https://doi.org/10.1002/hbm.25832>
- Muthukumaraswamy SD (2013). High-frequency brain activity and muscle artifacts
  in MEG/EEG: a review and recommendations. *Frontiers in Human Neuroscience*
  7:138. <https://doi.org/10.3389/fnhum.2013.00138>
- Onton J, Makeig S: data-length rule for ICA, as cited in Gabard-Durnam et al.
  (2018); not read directly.
- Pedroni A, Bahreini A, Langer N (2019). Automagic: standardized preprocessing
  of big EEG data. *NeuroImage* 200:460-473.
  <https://doi.org/10.1016/j.neuroimage.2019.06.046>
- Pion-Tonachini L, Kreutz-Delgado K, Makeig S (2019). ICLabel: an automated
  electroencephalographic independent component classifier, dataset, and
  website. *NeuroImage* 198:181-197.
  <https://doi.org/10.1016/j.neuroimage.2019.05.026>
- Tröndle M, Langer N (2026). Non-neural sources systematically impact aperiodic
  EEG activity. bioRxiv preprint. <https://doi.org/10.64898/2026.01.29.702285>

Software documentation used: MNE-BIDS-Pipeline 1.10.1 (`_config.py`:
`ica_l_freq`, `ica_h_freq`, `ica_n_components`, `ica_exclusion_thresholds`),
mne-icalabel 0.9.0 (`iclabel/features.py`, `iclabel/label_components.py`).
