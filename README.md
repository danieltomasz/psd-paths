# EEG PSD-PATHS Pipeline

**Version 0.1.0** - Automated batch processing pipeline for PATHS project EEG data

This is an automated EEG analysis pipeline for processing resting-state EEG data and extracting Power Spectral Density (PSD) features in the PATHS project. The project uses high-density EEG recordings (256-channel GSN-HydroCel montage) and processes them through filtering, artifact removal, ICA component extraction, and spectral parameterization using the `specparam` library.

## Features

- **Automated 3-stage pipeline** with template-based processing using Papermill
- **Parallel batch processing** with configurable job concurrency
- **Hybrid ICA strategy** with auto-suggestion and manual review capability
- **Comprehensive logging** with per-subject and pipeline-level reports
- **Reproducible workflows** using workspace-based package management

### Prerequisites

- **Python 3.13** (required) - [Download from python.org](https://www.python.org/downloads/)
- **Git** - to clone the repository

**Environment setup** - choose ONE of these options:

| Option | Tool | Best for |
|--------|------|----------|
| A (Recommended) | [uv](https://docs.astral.sh/uv/) | Fast setup, handles Python version automatically |
| B | Standard venv + pip | If you prefer familiar tools or can't install uv |

## Quick Start

### Option A: Using uv (Recommended)

1. **Install uv** (one-time setup):

```bash
# macOS
brew install uv
# or: curl -LsSf https://astral.sh/uv/install.sh | sh
```

1. Clone and setup:

Download the stable release (v0.1.0) from the `release-0.1` branch:

```bash
git clone --branch release-0.1 git@github.com:danieltomasz/psd-paths.git
cd psd-paths
uv sync
uv run python -m ipykernel install --user --name psd-paths-3.13
```

**Suggested:**

> [!TIP]
> If you have `make` installed (common on macOS/Linux), you can use shorthand commands defined in the [Makefile](Makefile):
>
> ```bash
> make sync    # runs uv sync
> make kernel  # installs the Jupyter kernel
> ```

### Option B: Using standard venv (no uv)

1. Ensure Python 3.13 is installed:

    ```bash
    python3 --version  # Should show 3.13.x
    ```

2. Clone and setup:

    ```bash
    git clone --branch release-0.1 git@github.com:danieltomasz/psd-paths.git
    cd psd-paths
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -e ".[dev]"
    pip install -e src/spectral
    python -m ipykernel install --user --name psd-paths-3.13
    ```

3. Verify Installation

    ```bash
    python -c "import mne; print('Success!')"
    ```

## Configuration

The **workspace** has two components:

1. **Root project** (`psd-paths`): Main analysis scripts and pipeline runner
2. **spectral package** (`src/spectral/`): Reusable EEG processing utilities (editable workspace dependency)

The project uses `settings.toml` for analysis parameters:

- `[paths]`: Project root and BIDS data paths
- `[preprocessing]`: Channels to remove (bad channels by design)
- `[experiment]`: Task name and parameters

## Pipeline Architecture

### Automated 3-Stage Processing

The pipeline uses **template-based notebooks** parameterized with Papermill:

1. **Stage 1** - `01_template-step1.ipynb`: Preprocessing
   - Bad channel removal
   - Bandpass and Notch filter application
   - Annotation of patch channels
   - Data quality checks

2. **Stage 2** - `02_Epochs.ipynb`: Epoching and PSD Analysis
   - Epoch extraction
   - Automatic bad epochs rejection

3. **Stage 3** - `03_Template_ICA_OLD_style.ipynb`: ICA and Spectral Parameterization
   - Independent Component Analysis with ICLabel
   - Bad ica component rejection
   - Spectral parameterization using `specparam`

### Hybrid ICA Strategy

- **Auto-suggest exclusions** with ICLabel for initial component classification
- **Manual review** capability for quality control
- **Re-entry from ICA**: Reprocess from ICA application onward without redoing expensive PyPREP/Autoreject
- **MNE Reports**: HTML QC reports, updateable after ICA re-application
- **Per-subject settings**: Parameters stored for full reproducibility

## Running the Pipeline

### Batch Processing

Use `templates/run_pipeline.py` to process multiple subjects in parallel:

```python
# Run all stages for all subjects
python templates/run_pipeline.py

# Configure in the script:
# - N_JOBS = 4              # Parallel job count
# - steps_to_run = [1,2,3]  # Which stages to execute
# - n_subjects = None       # Limit subject count (None = all)
```

The pipeline will:

- Discover subjects from BIDS directory automatically
- Execute notebook templates with subject-specific parameters
- Process subjects in parallel (configurable with `N_JOBS`)
- Generate processed notebooks in `outputs/pipeline/sub-{ID}/`
- Collect logs in `outputs/log/pipeline_{timestamp}.log`

### Pipeline Configuration

Edit `templates/run_pipeline.py` to customize:

```python
BIDS_ROOT = Path("/path/to/data/bids")           # Input data location
OUTPUT_ROOT = Path("/path/to/outputs/pipeline")  # Processed notebooks
OUTPUT_LOG = Path("/path/to/outputs/log")        # Log files
N_JOBS = 4                                       # Parallel workers
KERNEL_NAME = "psd-paths-3.13"                   # Jupyter kernel
```

### Output Organization

```
outputs/
├── pipeline/
│   └── sub-{ID}/
│       ├── sub-{ID}_step1-preprocessing.ipynb
│       ├── sub-{ID}_step1b-epochs-psd-analysis.ipynb
│       └── sub-{ID}_step2-psd-analysis.ipynb
└── log/
    └── pipeline_{timestamp}.log
```

## Project Structure

```
psd-paths/
├── templates/              # Pipeline notebook templates
│   ├── 01_template-step1.ipynb
│   ├── 02_Epochs.ipynb
│   ├── 03_Template_ICA_OLD_style.ipynb
│   └── run_pipeline.py    # Batch processing runner
├── src/spectral/          # Reusable EEG utilities (workspace package)
|-- data/
    |-- bids.              # data transformed into bids structure
    |___derricarives.      # Proudcts of analysis and processing
├── outputs/
    |---reports.           # Saved reports html files
│   ├── pipeline/          # Processed notebooks per subject
│   └── log/               # Pipeline execution logs
    |__ specparam          # Saved estimated specparam parameters  
├── notebooks/             # Development and exploration notebooks
├── settings.toml          # Analysis configuration
└── pyproject.toml         # Project dependencies and workspace config
```

## Version Information

- **Current Release**: v0.1.0 (stable baseline release)
- **Branch**: `release-0.1` (tagged snapshot for reproducibility)
- **Python**: >=3.13.11, <=3.14

View releases and tags at: <https://github.com/danieltomasz/psd-paths/releases>

## Export and Documentation

### Export notebooks as PDF (requires LaTeX)

Ensure LaTeX packages are installed:

```bash
tlmgr install titling
```

Convert notebook to PDF:

```python
pyenv activate psd-paths-3.13 && jupyter nbconvert --execute --to pdf notebook_path.ipynb
```

### Export Python code

```bash
files-to-prompt . -e py -e toml --cxml -o prompt-context.txt
```
