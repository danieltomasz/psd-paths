# Changelog

Changes to the PATHS resting-state analysis, the reasons for them, and the
results they affect. Parameter values are in `settings.toml`, with the reasons
in its comments. Changes to the processing code are listed in the
[eeg-spectral changelog](https://github.com/danieltomasz/eeg-spectral/blob/main/CHANGELOG.md).

## Runs

Each run writes its processed data and outputs to its own folder under
`runs/`. To reproduce a run, check out its tag and run `make sync`.

| Folder | Date | Tag | eeg-spectral | Python | Outcome |
| --- | --- | --- | --- | --- | --- |
| `runs/v0.3.0-dev` | 2026-10-02 | none (uncommitted, testing_branch) | 0.2.4 | 3.14.8 | 38/38 subjects, 190/190 steps, 17 min. 1-100 Hz, 1 Hz notch, 44 ICA components, muscle threshold 0.70. Test runs made the same day (1-100 Hz with the v0.2.3 ICA; 50 components; 6 Hz notch) were deleted; their results are in `docs/preprocessing-decisions.md`. |
| `runs/v0.2.3` | 2026-10-01 | v0.2.3 | 0.2.3 | 3.14.7 | 38/38 subjects, all 5 steps complete (190/190), 18 min. First run with the five-step pipeline and the recording-pause fix for all subjects. |
| `runs/v0.2.1` | 2026-10-01 | v0.2.1 | 0.2.1 (5 subjects: 0.2.2) | 3.13.15 (5 subjects: 3.14.7) | 38/38 subjects complete, 5 of them patched with the pause fix. The folder is no longer on disk. |
| `runs/v.0.1` | Aug 2026 | before v0.2.0 | local checkout | 3.13 | Made before the epoch fixes in eeg-spectral 0.2.0 (previously `data/derrivatives` and `outputs/`). Not used any more. |

If a run was patched instead of rerun, its folder contains a `PATCHES.md`.
`_test/`, `_rerun_czref/` and `runs/_check*/` are test runs and can be deleted.

## Unreleased

### Changed

- New document `docs/preprocessing-decisions.md`: the tests, numbers and
  literature behind the filter band, the notch, the ICA settings, eye artifact
  handling and the fit-error columns, with references.
- ICA (step 03): 44 components for every subject instead of the number
  explaining 99% of the variance capped at 40 (which gave 17 of 38 subjects in
  v0.2.3 fewer than 40, as few as 9). 44 is the largest number with at least
  30 unique samples per squared component in every subject (HAPPE's rule).
  ICLabel muscle threshold 0.70 instead of 0.80. Step 03 reports the samples
  per squared component and warns below 30. The component review widget runs
  only with `review = True` (in the 6 Hz test run it left sub-105 hanging in a
  pipeline run) and shows spectra up to 100 Hz. Needs eeg-spectral 0.2.4
  (`variance_threshold=None`). Reasons in `settings.toml` (`[cleaning]`) and
  `docs/preprocessing-decisions.md`. Together with the 1-100 Hz filter,
  `runs/v0.3.0-dev` compared with `runs/v0.2.3` (38 subjects, medians,
  measured channels only):
  - Components removed 6 -> 14.5 (muscle 1 -> 8, eye 2 -> 3); final epochs
    78 -> 76.5.
  - Exponent 1.03 -> 1.16 (higher in 31 subjects); the order of subjects is
    kept (Spearman rho 0.88 between runs). The change follows each subject's
    muscle activity (rho 0.80): sub-124 0.71 -> 1.35, sub-106 0.54 -> 0.94,
    sub-129 0.46 -> 0.71.
  - Exponent lower in 7 subjects, most in sub-109 (1.76 -> 1.19) and sub-137
    (1.80 -> 1.70), which had 15 and 9 components in v0.2.3; more eye and
    channel-noise components are now removed. sub-109's fit error 0.054 ->
    0.040.
  - Fit error 0.033 -> 0.032 (lower in 26 subjects). Subjects with good-fit
    share < 0.9: 6 -> 4. No subject marked Exclude (was sub-106 and sub-129;
    good-fit share 0.47 -> 0.65 and 0.46 -> 0.85).
  - Offset nearly unchanged (rho 0.93). Alpha peak frequency 8.87 -> 9.00 Hz
    (rho 0.99), peak power above the aperiodic fit 0.44 -> 0.49.
  - Exponent vs age: Spearman -0.38 -> -0.44.
- Step 01 filters 1-100 Hz instead of 1-40 Hz, and the 50/100 Hz notch is
  1 Hz wide instead of mne's default 0.25 Hz. A 6 Hz notch (47-53 Hz, as in
  RELAX) was tested and dropped: it removes the weak line-noise skirt but
  leaves a gap in every ICA component's spectrum, and ICLabel then calls muscle
  components "other" (on the same components, muscle p > 0.7: 350 with 1 Hz,
  261 with 2 Hz, 71 with 6 Hz). 1.5 and 2 Hz removed hardly more line noise
  than 1 Hz and found 13% and 28% fewer muscle components. The 40 Hz low-pass was there
  because the notch did not remove the line noise: with the 0.25 Hz notch, 14
  of 38 subjects kept channels with 50 Hz more than 6 dB above 45-47/53-55 Hz
  (sub-108: 185 channels). With 1 Hz, no channel in any subject (worst
  subject, 95th percentile channel: +1.2 dB; 100 Hz removed too). The data now
  matches what ICLabel was trained on (1-100 Hz, Pion-Tonachini et al. 2019),
  and muscle activity above 40 Hz stays visible to ICA and to the quality
  checks. specparam (2-35 Hz) does not use anything above 40 Hz. Bad channels
  (pyprep, LOF) and autoreject now work on 1-100 Hz data, so both change for
  all subjects; `lof_threshold` and `fallback_ptp_uv` were set on 1-40 Hz
  data. Needs eeg-spectral 0.2.4 (`notch_widths` in `raw_filtered`).
  Diagnostic PSD plots in steps 01 and 02 now go to 110 and 100 Hz.
  Tested in a run with only this change (ICA as in v0.2.3; the folder was
  deleted), compared with `runs/v0.2.3` (38 subjects):
  - Bad channels (step 01): median 9.5 -> 9, most subjects within 1-3
    channels; the largest change is sub-108, 13 -> 8.
  - Autoreject before ICA (step 02, which fits on a 125 Hz copy): median
    peak-to-peak amplitude 43 -> 51 uV, rejected epochs median 6.5 -> 6.5
    (6 subjects more, 15 fewer, 17 the same; at most 5 epochs). Autoreject sets
    its thresholds from each recording, so they rise with the amplitude.
  - Autoreject after ICA (step 04, 1-100 Hz): median 0 rejected in both runs;
    sub-114 1 -> 6, otherwise at most 2 more. Final epochs median 78 -> 77.5.
  - Neither change in rejections is related to how much muscle activity a
    subject has (Spearman rho 0.13 and -0.10).
  - ICA (same thresholds): ICLabel now finds muscle components. Removed
    components median 6 -> 12, muscle 1 -> 5.
  - specparam: median exponent 1.03 -> 1.14 (32 of 38 subjects higher); no
    subject marked Exclude (was sub-106 and sub-129: good-fit share 0.47 ->
    0.69 and 0.44 -> 0.53; sub-129 is just above the 0.5 limit).
- Group report: two columns with the specparam fit error, next to the R^2
  rule and not used by it (Keep/Exclude is unchanged).
  `specparam_error_median` is the median over measured channels (interpolated
  ones left out) of specparam's mean absolute fit error, in log10 power.
  `specparam_error_z` is how far that is from the other subjects of the run
  (robust z-score: median and MAD). Reason: R^2 rises with the exponent, because
  a steeper spectrum leaves more variance to explain, while the error does not
  (across channels, Spearman rho 0.63 vs -0.11). A subject with a steep spectrum
  can therefore pass the R^2 rule with poor fits. In the filter-only test run:
  sub-109 z = 3.1 with good-fit share 1.00. In `runs/v0.3.0-dev`: largest
  sub-112 and sub-127 (z = 2.1, error 0.045); group median error 0.032.
- Step 01 no longer uses Hamilton. The notebook calls the preprocessing
  functions from eeg-spectral (`spectral.flows.preprocessing`) one after the
  other: load, drop channels, resample/notch/filter/crop, detect bad channels,
  mark them, average reference, save, then the report and plots. The functions
  themselves are unchanged. Checked on sub-101, 109, 114 and 145 against
  `runs/v0.2.3`: identical step 01 data, bad channels and annotations, identical
  final epochs and specparam output. `sf-hamilton` is no longer a direct
  dependency of the project (eeg-spectral still uses it).
- Group step: `templates/SpecparamTogether.ipynb` (moved from `notebooks/`)
  collects the specparam results of all subjects of a run and writes
  `all_subjects_specparam.csv`, `unique_specparam_results.csv` (one row per
  channel) and `participant_exclusion_report.csv` to `<outputs_root>/group/`.
  It runs once at the end of `run_pipeline.py`, with `make group`, or on its
  own; it reads the run folder from `settings.toml` instead of a fixed path and
  only reads `*-specparam.csv` files. The exclusion rule is unchanged (fewer
  than 50% of channels with R^2 >= 0.9 means Exclude) and is now set in
  `[group]` in `settings.toml`. In the report, the columns from the specparam
  fits are prefixed `specparam_` (`specparam_n_channels`,
  `specparam_good_fit_share`, `specparam_status`; the old names were
  `total_channels`, `ratio_good_channels`, `status`), followed by the step 04
  preprocessing counts (interpolated channels, ICA components, epochs). The
  report also lists subjects without results and failed steps. Both channel
  tables have an `interpolated` column (True for channels marked bad and
  interpolated in step 04), taken from each subject's step 04 record and checked
  against its count (`runs/v0.2.3`: 420 channels in 38 subjects).
  For `runs/v0.2.3`: 38 subjects, sub-106 and sub-129 marked Exclude.
- `make` on its own lists the commands; `make run` runs the whole pipeline
  (all subjects, then the group step). Removed `make test`, which pointed to a
  `tests/` folder that does not exist.
- README rewritten for the current pipeline: setup, which command to use when,
  the five steps and the group step with their outputs, the run folder layout,
  manual ICA review, and how to reproduce a run.
- The earlier group tables in `notebooks/` are from the December 2025 run
  (tag v0.1.0): 36 subjects (sub-114 and sub-127 missing), made before all
  later fixes. sub-128 was excluded there because of its recording pause.
- Noted during that check: the ICA weights of one subject (sub-101) differed
  from `runs/v0.2.3` in the 7th significant digit, so its ICA fingerprint
  changed; the removed components and the final data were the same. This comes
  from multithreaded computation in the ICA fit, not from the step 01 change.

## 0.2.3 - 2026-10-01

### Data issues

Five recordings contain one acquisition pause (EGI "acquisition skip"). MNE
fills the pause with zeros and marks it as `BAD_ACQ_SKIP`. The pauses are also
listed in each subject's `events.tsv`.

| Subject | Pause starts | Length |
| --- | --- | --- |
| sub-114 | 314.8 s | 43.5 s |
| sub-124 | 5.6 s | 7.3 s |
| sub-127 | 65.5 s | 15.5 s |
| sub-128 | 211.9 s | 52.3 s |
| sub-139 | 5.4 s | 6.1 s |

Epoching used `reject_by_annotation=False`, leaving bad segments to
autoreject. The pauses therefore became flat 5 s epochs, which autoreject does
not reject. Effect on `runs/v0.2.1`:

- sub-114 and sub-128: in step 5, the second autoreject pass kept only the
  pause epochs (11 and 14). The spectrum was zero and specparam failed.
- sub-124, sub-127, sub-139: 1, 3 and 1 pause epochs were in the final
  average. Power was 1-3.5% lower, and the aperiodic offset about 0.005-0.016
  lower (log10). Exponent and peaks were not affected.
- Epochs at the start or end of a pause contain large transients (up to about
  3000 uV). Autoreject rejected all of them, so they did not affect results.

Test of the fix on these 5 subjects: without the pause epochs, step 5 keeps 158
epochs for sub-114 and 76 for sub-128, instead of none. The flat epochs had
caused the second autoreject pass to reject all other epochs. For sub-124, 127
and 139 the number of kept epochs also changes (70, 80, 69 instead of 68, 85,
84), because the flat epochs no longer affect the autoreject thresholds.

### Fixed

- eeg-spectral 0.2.2 drops all epochs that overlap `BAD_ACQ_SKIP`, including
  the epochs at the start and end of a pause. Subjects without a pause are not
  affected. `02_Epochs` prints the number of dropped epochs. The 5 affected
  subjects were rerun and copied into `runs/v0.2.1`.

### Changed

- Python 3.14.7 (`requires-python = ">=3.14.7,<3.15"`). The package versions in
  `uv.lock` are the same as before.
- eeg-spectral 0.2.3. The list of bad channels is now sorted, so it is the same
  in every run; before, only its order could differ. The data does not change.
- Five steps instead of six: `04_ICA_apply` and `05_Interpolate` are merged
  into `04_ICA_apply_interpolate`, and `06_Specparam` is now `05_Specparam`.
  They were split so that a changed ICA selection could be applied without
  rerunning the second autoreject pass, but that pass takes under a minute and
  has to be rerun after a new ICA selection anyway. The cleaned epochs are still
  saved (`clean_ica-epo.fif`) and read back before autoreject, so autoreject
  gets the same 32-bit data as before. The steps are now listed in
  `templates/run_pipeline.py`, which `run_december.py` and the dashboard also
  use. Runs made before this change (`runs/v0.2.1`) have six steps: in their
  status files, step 5 is Interpolate and step 6 is Specparam.
- `make status` works again (it called `show_status()` without the config).
  `make status`, `make dashboard`, `make test` and `make kernel` use the locked
  environment (`uv run --locked`); before, they could update `uv.lock` and the
  installed packages without saying so.
- Templates 03-05 (now 03 and 04) restructured following the ds004504 templates. Results are
  the same: checked on sub-101 and sub-109 (same ICA decomposition and removed
  components, identical epochs after each step, identical specparam output).
  - 03 shows one overview of all components (ICLabel class, probability, share
    of variance; removed components in red) instead of plotting all ~40
    components several times. An optional review widget at the end saves a
    manual selection to the decision file.
  - ICA apply no longer loads and plots the raw data. The before/after overlay is
    added to the report.
  - The second autoreject pass shows its result and the channels with the most
    rejections.
  - 02-05 write to one report per subject (`sub-XXX_report.html`, one section
    per step) instead of one file per step. The PSD comparison of rejected and
    kept epochs from 02 is now included in the report.
  - Settings are read without default values in the code, so a missing key
    gives an error. The correlation threshold for matching a manual ICA
    selection after a refit is now in `settings.toml` (`ica_corr_threshold =
    0.95`, same value).
  - Unused code and imports removed.

### Open methods questions

Not changed yet, because each of these changes results:

- ICLabel is applied to data filtered at 1-40 Hz. It was trained on 1-100 Hz
  data and gives a warning for every subject. Muscle components are the most
  likely to be missed. Possible change: fit ICA and run ICLabel on a 1-100 Hz
  copy and apply the result to the 1-40 Hz data.
- The ECG correlation in 03 uses the continuous recording, including segments
  that were not used for the ICA fit. ICLabel uses the epochs the ICA was fitted
  on. Using the epochs for both would be consistent.
- 03 removes another 3 s at each end of the recording, after 01 already did.
  Only the ECG correlation uses this data.
- Muscle threshold (checked 2026-10-01 on `runs/v0.2.3`, sub-106, 129, 101,
  107, without the second autoreject pass). The two excluded subjects differ:
  - sub-106: flat spectra only at the lower rim and sides of the head (rim
    exponent 0.17, top 1.19), 10 components labelled muscle of which 9 are kept
    because their probability is below 0.80. Also removing muscle components
    with p >= 0.5 gives rim exponent 0.89, 81% well-fitted channels instead of
    48%, exponent 0.98 instead of 0.56. The exclusion is caused by remaining
    muscle activity and goes away with better muscle removal.
  - sub-129: flat spectra over the whole head, half the usual theta and alpha
    power, 17 interpolated channels, 7 brain components. Removing muscle
    components hardly helps (49% instead of 47%). This looks like a recording
    quality problem; exclusion is defensible on that ground, not on R^2 alone.
  - Normal subjects hardly change (sub-101 identical, sub-107 exponent 1.56 to
    1.63 with one more muscle component removed).
- The R^2-based exclusion rule (`specparam_good_fit_share`) follows the
  exponent (Spearman rho = 0.76 across subjects), so it tends to exclude
  subjects with flat spectra. An error-based criterion or a check of the
  reason for each exclusion would avoid that.

## 0.2.1 - 2026-10-01

### Changed

- One folder per run: `derivatives_root` and `outputs_root` in `settings.toml`
  both point to `runs/<version>/`, so a run can be archived, shared or deleted
  in one piece. `runs/` is not tracked by git.
- `generate_dashboard.py` reads its paths from `settings.toml` in the same way
  as the runner. Before, it read from a folder literally named `{outputs}`.

### Runs

- `runs/v0.2.1`: all 38 subjects, steps 1-6, 16 min. See the table above.

## 0.2.0 - 2026-10-01

### Changed

- eeg-spectral is installed from its GitHub tag (now 0.2.1) instead of a local
  copy, so `uv.lock` records the exact commit. This includes the eeg-spectral
  0.2.0 fixes, which change results:
  - Epochs were shifted by 3 s because the crop at the start of step 1 was not
    taken into account. The first epoch was dropped and the last seconds of the
    recording were not used.
  - Epochs were one sample too long (1251 instead of 1250 samples at 250 Hz).
  - The removed ICA components are recorded per subject and checked against the
    fitted decomposition.

  All results made before this version differ and have to be rerun.
- `task = "RESTING"` in `settings.toml`, the task label in the BIDS file names.
  eeg-spectral 0.2 uses it to find the recording. The previous value `"rest"`
  did not match any file.
- `make sync` installs exactly the versions in `uv.lock`. `make update` (after
  editing `pyproject.toml`) and `make upgrade` (newer packages; rerun the
  pipeline afterwards) change the lock file.
- Notebook outputs are removed by a git filter instead of a pre-commit hook.
  Open notebooks keep their outputs.
- Removed the deprecated Hamilton telemetry call from step 1.

## Analysis decisions, 2026-08 to 2026-09

The reasons are given in the comments in `settings.toml`.

- 2026-08-28, bad channels: LOF detection added in addition to PyPREP/RANSAC
  (`use_lof`, 20 neighbours, threshold 4.0). RANSAC can miss clusters of
  neighbouring bad electrodes, e.g. E221-E223 in sub-109 (Kumaravel et al.
  2022).
- 2026-08-28, epoch rejection: if autoreject's threshold search fails, a fixed
  peak-to-peak threshold of 150 uV is used (at most 25% of epochs rejected).
- 2026-08-28, epochs: overlap corrected to 1.5 s in the settings. They said
  3.0 s, which never matched the code.
- 2026-09-30, ICA: ICLabel thresholds per class (eye blink 0.50, others 0.80).
  With 0.80 for all classes, 6 of 20 subjects kept blink components.
- 2026-09-30, ICA: components correlated with the ECG are not removed if
  ICLabel classifies them as brain. sub-107 had lost 25% of its alpha power to
  two such components.
- 2026-09-30, step 5: if the second autoreject pass leaves fewer than 10
  epochs, the epochs from before this pass are kept. Added after sub-114 lost
  all its epochs; in `runs/v0.2.1` this was caused by the recording pause (see
  0.2.2). The check is kept.

## 0.1.0 - 2025-12-19

- First automated version of the pipeline (Papermill templates, batch runner).
