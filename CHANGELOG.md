# Changelog

What changed in the PATHS resting-state analysis, why, and which results each
change affects. Parameter values live in `settings.toml` (with the reasoning in
its comments); changes to the processing code itself are in the
[eeg-spectral changelog](https://github.com/danieltomasz/eeg-spectral/blob/main/CHANGELOG.md).

## Runs

Each run writes everything (processed data and outputs) into its own folder
under `runs/`. To reproduce one, check out its tag and run `make sync`.

| Folder | Date | Tag | eeg-spectral | Python | Outcome |
| --- | --- | --- | --- | --- | --- |
| `runs/v0.2.2` | planned | v0.2.2 | 0.2.2 | 3.14.7 | Rerun of all 38 subjects after the pause fix. |
| `runs/v0.2.1` | 2026-10-01 | v0.2.1 | 0.2.1 | 3.13.15 | 36/38 complete. sub-114 and sub-128 failed at specparam; sub-124, 127 and 139 slightly off. Superseded, see 0.2.2. |
| `data/derrivatives`, `outputs/` | Aug 2026 | before v0.2.0 | local checkout | 3.13 | Made before the epoch fixes in eeg-spectral 0.2.0. Superseded. |

`_test/` and `_rerun_czref/` are throwaway test runs.

## Unreleased (0.2.2)

### Data issues

Five recordings contain one acquisition pause (EGI "acquisition skip"). MNE
fills a pause with zeros and marks it `BAD_ACQ_SKIP` (also listed in each
subject's `events.tsv`):

| Subject | Pause starts | Length |
| --- | --- | --- |
| sub-114 | 314.8 s | 43.5 s |
| sub-124 | 5.6 s | 7.3 s |
| sub-127 | 65.5 s | 15.5 s |
| sub-128 | 211.9 s | 52.3 s |
| sub-139 | 5.4 s | 6.1 s |

Epoching used `reject_by_annotation=False` (bad segments are left to
autoreject), so the zero-filled pause became flat 5 s epochs, and autoreject
never rejects flat epochs. Effect on `runs/v0.2.1`:

- sub-114 and sub-128: the second autoreject pass (step 5) kept only the pause
  epochs (11 and 14), so the spectrum was all zeros and specparam failed.
- sub-124, sub-127, sub-139: 1, 3 and 1 pause epochs stayed in the final
  average, lowering power by 1-3.5% (aperiodic offset about 0.005-0.016 lower
  in log10). Exponent and peaks are not affected.
- Epochs straddling a pause edge (stop/start transients up to ~3000 uV) were
  all rejected by autoreject, so they did not reach the results.

With the pause epochs removed (test on these 5 subjects), step 5 keeps 158
real epochs for sub-114 and 76 for sub-128 instead of none: the flat epochs
had made the second autoreject pass reject every real epoch. For sub-124, 127
and 139 the number of kept epochs changes too (70, 80, 69 instead of 68, 85,
84), because autoreject no longer sets its thresholds with flat epochs in the
data.

### Fixed

- eeg-spectral 0.2.2: epochs overlapping `BAD_ACQ_SKIP` are always dropped,
  including the ones straddling a pause edge. The other subjects are unchanged.
  `02_Epochs` prints how many epochs were dropped for this.

### Changed

- Python 3.14.7 (`requires-python = ">=3.14.7,<3.15"`). All package versions
  in `uv.lock` stay the same.

## 0.2.1 - 2026-10-01

### Changed

- One folder per run: `derivatives_root` and `outputs_root` in `settings.toml`
  both point into `runs/<version>/`, so a run can be archived, shared or deleted
  as a unit. `runs/` is not tracked by git.
- `generate_dashboard.py` takes its paths from `settings.toml` the same way the
  runner does (it used to read a literal `{outputs}` folder).

### Runs

- `runs/v0.2.1`: all 38 subjects, steps 1-6, 16 min. See the table above.

## 0.2.0 - 2026-10-01

### Changed

- eeg-spectral is installed from its GitHub tag (now 0.2.1) instead of a local
  checkout, so `uv.lock` records the exact commit. With it come the eeg-spectral
  0.2.0 fixes that change results:
  - Epochs were placed 3 s off (the crop at the start of step 1 was ignored),
    so the first epoch was dropped and the last seconds were never used.
  - Epochs were one sample too long (1251 instead of 1250 at 250 Hz).
  - ICA exclusions are recorded per subject and checked against the fitted
    decomposition.

  Every result made before this version differs and must be rerun.
- `task = "RESTING"` in `settings.toml`, the label in the BIDS file names.
  eeg-spectral 0.2 uses it to pick the recording; the old placeholder `"rest"`
  matched nothing.
- `make sync` installs exactly what `uv.lock` pins. `make update` (after editing
  `pyproject.toml`) and `make upgrade` (newer packages; rerun afterwards) are
  the only ways to change it.
- Notebook outputs are stripped by a git filter instead of a pre-commit hook,
  so open notebooks keep their outputs.
- Removed the deprecated Hamilton telemetry call from step 1.

## Analysis decisions, 2026-08 to 2026-09

Recorded in `settings.toml` (comments give the full reasoning):

- 2026-08-28, bad channels: LOF detection added next to PyPREP/RANSAC
  (`use_lof`, 20 neighbours, threshold 4.0). RANSAC misses contiguous clusters
  of failed electrodes, e.g. E221-E223 in sub-109 (Kumaravel et al. 2022).
- 2026-08-28, epoch rejection: when autoreject's threshold search degenerates,
  fall back to a fixed 150 uV peak-to-peak criterion (at most 25% of epochs).
- 2026-08-28, epochs: overlap documented as 1.5 s. The settings had claimed
  3.0 s, which never matched the code.
- 2026-09-30, ICA: ICLabel thresholds per class (eye blink 0.50, others 0.80).
  At a flat 0.80, 6 of 20 subjects kept blink components.
- 2026-09-30, ICA: ECG-correlated components that ICLabel classifies as brain
  are kept. sub-107 had lost 25% of its alpha power to two of them.
- 2026-09-30, step 5: if the second autoreject pass leaves fewer than 10
  epochs, the uncleaned epochs are kept (sub-114 had lost all of them). In
  `runs/v0.2.1` that collapse turned out to be caused by the recording pause
  (see 0.2.2); the guard stays as a safety net.

## 0.1.0 - 2025-12-19

- First automated version of the pipeline (Papermill templates, batch runner).
