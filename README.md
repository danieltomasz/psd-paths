# EEG PSD-PATHS pipeline

Code  used for analysis of resting state EEG data and extracting PSD features.

## Installation

Download the files with

```bash
git clone --depth 1 --branch pipeline git@github.com:danieltomasz/psd-paths.git
```

then go to the folder you download  the repo and open terminal in it.


Assuming that you have conda installed, run the command below to install the conda environment with all necessary packages for the analysis (you need to have conda installed, you can name your environmen differently than `conda-paths-3.12`):

If you are running computer  with Apple Sillicon processor you can use Apple Accelerate fromework for  faster numeric computations: 

```bash
conda env create -n conda-paths-3.12 -f local.yml
```

If you are running computer with Intel processor you can use Intel MKL for faster numeric computations (modify the yml file accordingly):

```bash
conda env create -n conda-paths-3.12 -f local_noARM.yml
```

## Analysis

All notebooks are located in the `analysis` folder. The analysis is organized in a way that you can run each notebook independently, but they are also designed to be run sequentially.

In the settings.toml file you can define the parameters for the analysis, such as the channels to remove, filter parameters, etc. You should also define your local path to the project, it will be used later by some functions to find the data and save the results.

### 0. Copy .mff files to the project folder

There is a notebook called `EnsureFolderStructure.ipynb` which will copy unstructured  'mff' files into subfolder per subject called . Define paths of input and output folders and run cell manually. If you have already copied the files, you can skip this step.

### 1. Preprocessing

This step is done in the notebook `sub-101-raw-notch.ipynb`. It will load the raw data, apply notch filter, remove bad channels, and save the preprocessed data. The preprocessed data will be saved as epochs in the  `data/epochs` folder.

### 2. ICA and Extracting PSD features

This step is done in the notebook `sub-101-ica.ipynb`. It will load the preprocessed data, run ICA, and extract the spectral features. The features will be saved in the `outputs/specparam` folder.

For every subject, the PSD features will be saved in a separate folder named after the subject (e.g., `sub-101_1`).

## Running analysis for many sunbjects

You duplicate the notebooks for each subject to run the same analysis, you just need to change the subject ID in the notebook name and in the code. The strucure of the folders will be the same for each subject.

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
