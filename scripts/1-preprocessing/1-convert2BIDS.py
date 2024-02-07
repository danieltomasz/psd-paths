

# %%
import mne
import toml
import mne_bids
import re
from pathlib import Path


config = toml.load("../../.env.toml")

project_path = config["project"]["path"]
output_path = f"{project_path}/data/raw/BIDS" #Into which directory the data should go


def convert2bids(path, sub, task='rest', test = True):
    raw = mne.io.read_raw_egi(path, preload=True ).resample(sfreq=250)
    raw.set_montage('GSN-HydroCel-257', match_alias= {'VREF' :'Cz'})
    #raw.plot_sensors(show_names=True);
    bids_path = mne_bids.BIDSPath(subject=sub,
                                session='01',
                                task= task,
                                datatype='eeg',
                                root=output_path )
    mne_bids.write_raw_bids(raw, bids_path=bids_path,  format='EEGLAB',allow_preload=True,
                            overwrite=True);


# %% Run the conversion in a loop
# Define the directory
dir_path = Path("/Volumes/ExtremePro/DATA PATHS/DATI_PATHS/Raw_data/")

# List all files in the directory
files = [file for file in dir_path.glob('*.mff')]

for fname in files:
    # Use regex to find the first number after "PATHS"
    match = re.search(r'(?<=PATHS_)(\d+)|(?<=PATHS_)(\d+)', fname.stem)
    if match:
        sub = match.group(1)
    convert2bids(fname, sub)
    print(f"Converted {fname} to BIDS format")

# %%
