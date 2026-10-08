# PATHS resting-state EEG pipeline

Processes the PATHS resting-state recordings (EGI, 256 channels, BIDS format)
subject by subject: preprocessing, epoching, ICA cleaning and spectral
parameterisation (specparam), followed by group tables of the results. Each step
is a Jupyter notebook in `templates/`, run for every subject with papermill. The
processing functions are in a separate package,
[eeg-spectral](https://github.com/danieltomasz/eeg-spectral), installed at a
fixed version. Changes and analysis decisions are recorded in
[CHANGELOG.md](CHANGELOG.md); the tests and literature behind the filtering,
line-noise and ICA settings are in
[docs/preprocessing-decisions.md](docs/preprocessing-decisions.md).

## Setup

You need git, [uv](https://docs.astral.sh/uv/) (it installs the right Python
version itself), make (macOS/Linux), and read access to the psd-paths and
eeg-spectral repositories on GitHub.

```bash
git clone git@github.com:danieltomasz/psd-paths.git
cd psd-paths
git checkout v0.2.3   # the version to run or reproduce, see CHANGELOG.md
make sync             # installs the exact package versions in uv.lock
make kernel           # registers the Jupyter kernel (once per machine)
```

Then:

- put the BIDS dataset in `data/bids/` (the data is not in git);
- in `settings.toml`, set `project_root` to the folder you cloned into.

`make` on its own lists all commands.

## Running

| Command | What it does |
| --- | --- |
| `make run` | Runs all subjects through steps 01-05, then the group step. This is the normal way to run the pipeline. |
| `make status` | Shows which steps succeeded or failed for each subject. |
| `make group` | Rebuilds the group tables. Only needed after a partial rerun (some subjects or steps); `make run` already does it. |
| `make dashboard` | Writes an HTML overview of the run (`dashboard.html` in the run's outputs). |

Partial reruns are done from Python, started in the project folder with
`uv run python`:

```python
import sys; sys.path.insert(0, "templates")
from run_pipeline import cfg, run_group
from spectral.runner import run_pipeline, rerun_failed

run_pipeline(cfg, subjects_to_run=["101", "127"], steps_to_run=[4, 5])
rerun_failed(cfg)   # only the steps that failed
run_group(cfg)      # then update the group tables (same as make group)
```

Steps that depend on each other have to be rerun together: a change in step 01
means rerunning 01-05 for that subject.

## Steps

| Step | Notebook | What it does | Main output (per subject) |
| --- | --- | --- | --- |
| 01 | `01_Preprocessing` | Load the recording, drop unused channels, resample to 250 Hz, notch 50/100 Hz (1 Hz wide), filter 1-100 Hz, detect bad channels (pyprep and LOF), average reference | `derivatives/processed/sub-XXX/sub-XXX_annotated_filtered_raw.fif` |
| 02 | `02_Epochs` | 5 s epochs (1.5 s overlap), drop epochs in recording pauses, reject bad epochs with autoreject | `derivatives/epochs/sub-XXX/sub-XXX_good_epochs-epo.fif` |
| 03 | `03_ICA_fit` | ICA, ICLabel classification, automatic selection of the components to remove, optional manual review | `derivatives/analysis/sub-XXX/sub-XXX_ica-decision.json` |
| 04 | `04_ICA_apply_interpolate` | Remove the selected components, second autoreject pass, interpolate bad channels | `derivatives/analysis/sub-XXX/sub-XXX_interpolated-epo.fif`, `outputs/specparam/sub-XXX/sub-XXX_ica_metadata.csv` |
| 05 | `05_Specparam` | Fit specparam (2-35 Hz) on every channel | `outputs/specparam/sub-XXX/sub-XXX-specparam.csv` |
| group | `SpecparamTogether` | Combine all subjects, flag subjects with poor fits | `outputs/group/*.csv` |

Every step also adds a section to one HTML report per subject
(`outputs/reports/sub-XXX/sub-XXX_report.html`), and the executed notebook of
every step is kept (`outputs/pipeline/sub-XXX/`).

## Where the results go

Each run writes everything to its own folder, set by `derivatives_root` and
`outputs_root` in `settings.toml`:

```
runs/v0.2.3/
├── derivatives/        processed/, epochs/, ica/, analysis/ (per subject)
└── outputs/
    ├── pipeline/       executed notebooks, per subject, and the group notebook
    ├── reports/        one HTML report per subject
    ├── specparam/      specparam results and step 04 counts, per subject
    ├── group/          all_subjects_specparam.csv, unique_specparam_results.csv,
    │                   participant_exclusion_report.csv
    ├── figures/  log/  pipeline_status.json  dashboard.html
```

For a new run, change the folder name in both lines (e.g. `runs/v0.3.0`), so
earlier runs are not overwritten. `runs/` is not tracked by git.

## Settings

All parameters are in `settings.toml`, with the reasons for the values in its
comments: `[paths]` (data and run folders), `[pipeline]` (parallel jobs, Jupyter
kernel), `[experiment]` (BIDS task and session), `[preprocessing]`, `[epochs]`,
`[cleaning]` (ICA), `[interpolate]` (second autoreject pass) and `[group]`
(exclusion criteria).

## Manual ICA review

1. Open `templates/03_ICA_fit.ipynb`, set `subject_id` and `review = True` in
   the parameter cell, and run it. (The pipeline runs it with `review = False`,
   so the widget is skipped there.)
2. In the last section, go through the components, mark the ones to remove and
   press "Save decision". The choice is saved in the decision file next to the
   automatic one.
3. Rerun steps 04 and 05 for that subject, then the group step:
   `run_pipeline(cfg, subjects_to_run=["101"], steps_to_run=[4, 5])` and
   `make group`.

## Reproducing a run and changing the code

- Each run folder belongs to a git tag (table in [CHANGELOG.md](CHANGELOG.md)).
  To reproduce it: `git checkout <tag>`, `make sync`, `make run`.
- The eeg-spectral version is set by its tag in `pyproject.toml`. To use a new
  version, change the tag, run `make update`, and start a new run folder.
- `make sync` never changes package versions; `make update` and `make upgrade`
  do, and are recorded in `uv.lock`.

## Notebooks and git

Notebook outputs are removed when notebooks are committed (nbstripout as a git
filter), so the templates stay clean while open notebooks keep their outputs.
In a new clone, set this up once if you will commit notebooks:

```bash
uv run nbstripout --install
git config filter.nbstripout.extrakeys metadata.kernelspec
```

## Exporting a notebook as PDF

Needs LaTeX (`tlmgr install titling`):

```bash
uv run jupyter nbconvert --to pdf runs/v0.2.3/outputs/pipeline/sub-101/sub-101_05_Specparam.ipynb
```
