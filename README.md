# EEG PSD-PATHS pipeline

Code  used for analysis of restring state EEG data and extracting PSD features.

## Installation
Download the files with

```
git clone --depth 1 --branch pipeline git@github.com:danieltomasz/psd-paths.git
```
then go to the folder you download  the repo and open terminal in it.


Assuming that you have conda installed, run the command below to install the conda environment with all necessary packages for the analysis (you need to have conda installed, you can name it differently than `conda-paths-3.12`):

```bash
conda env create -n conda-paths-3.12 -f local_noARM.yml
```
If you are runninhg Apple Sillivon computer and can use Apple Sillicon framerwork use: 


```bash
conda env create -n conda-paths-3.12 -f local.yml
```




## How to use scripts and notebooks for analysis of new subjects

1. Ensure that the folders with the  subjects adhere to specific conventions, 

There is a notebook that will copy unstructurized 'mff' files into supfolders pers subject called `analysis/EnsureFolderStructure.ipynb`. Define paths of input and output folders and run cell manually.




## Install python packages manually

```bash
python3 -m venv evn_name
source evn_name/bin/activate
pip3 install -r requirement.txt
```

# Analysis steps


# Export as PDF 

ensure latex packaeges are installed

```bash
tlmgr install titling
```

```python
pyenv activate psd-paths-3.13 && jupyter nbconvert --execute --to pdf analysis/preprocessing/sub-101_1-raw-notch.ipynb
```

export python code 

```bash
files-to-prompt . -e py -e toml --cxml -o prompt-context.txt
```
