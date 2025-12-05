# EEG PSD-PATHS pipeline

This is an EEG analysis pipeline for processing resting-state EEG data and extracting Power Spectral Density (PSD) features in the PATHS project. The project uses high-density EEG recordings (256-channel GSN-HydroCel montage) and processes them through filtering, artifact removal, ICA, and spectral parameterization using the `specparam' library.

## Installation

Download the files with

```bash
git clone --depth 1 --branch pipeline git@github.com:danieltomasz/psd-paths.git
```

then go to the folder you download  the repo and open terminal in it.

This version of code uses `uv` package for dowanloading and managing packages.

If you installed `uv` you can run `uv sync` in the folder or `make sync` to include also decelopmental dependencies.

The **uv workspace** has two components:

1. **Root project** (`psd-paths`): Main analysis scripts and `notebooks/` directory
2. **spectral package** (`src/spectral/`): Reusable EEG processing utilities

The `spectral` package is installed in editable mode via workspace configuration in `pyproject.toml`.

## Analysis

All notebooks are located in the `analysis` folder. The analysis is organized in a way that you can run each notebook independently, but they are also designed to be run sequentially.

### 0. Copy .mff files to the project folder

There is a notebook called `EnsureFolderStructure.ipynb` which will copy unstructured  'mff' files into subfolder per subject called . Define paths of input and output folders and run cell manually. If you have already copied the files, you can skip this step.

In the settings.toml file you can define the parameters for the analysis, such as the channels to remove, filter parameters, etc. You should also define your local path to the project, it will be used later by some functions to find the data and save the results.

### 1. Preprocessing

This step is done in the notebook `sub-101-step-1-filter-raw-notch.ipynb`. It will load the raw data, apply notch filter, remove bad channels, and save the preprocessed data. The preprocessed data will be saved as epochs in the  `data/epochs` folder.

### 2. ICA and Extracting PSD features

This step is done in the notebook `sub-101-step_2-ica-specparam.ipynb`. It will load the preprocessed data, run ICA, and extract the spectral features. The features will be saved in the `outputs/specparam` folder.

## Running analysis for many sunbjects

You  can duplicate notebooks for each subject to run the same analysis, you just need to change the subject ID in the notebook name and in the code. The structure of the folders will be the same for each subject.

## Export notebooks as PDF (requires LaTeX installed)

Ensure latex packaeges are installed

```bash
tlmgr install titling
```

```python
pyenv activate psd-paths-3.13 && jupyter nbconvert --execute --to pdf notebook_path.ipynb
```

## export python code

```bash
files-to-prompt . -e py -e toml --cxml -o prompt-context.txt
```
