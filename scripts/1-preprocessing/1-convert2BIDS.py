
"""
This script converts EEG data in MFF format to BIDS format.

Author: Daniel Borek
Date: 2024-03-12
"""
# %%
from pathlib import Path
import re
import mne
import toml
import mne_bids

config = toml.load("../../.env.toml")

project_path = config["project"]["path"]
dir_path = config["project"]["raw_dir_path"] #Where the data is stored

output_path = f"{project_path}/data/raw/BIDS" #Copy the data to this location
Path(output_path).mkdir(parents=True, exist_ok=True)

def convert2bids(path, sub, task='rest', test=True):
    """
    Use MNE-BIDS to convert the EEG data to BIDS format.
    """
    raw = mne.io.read_raw_egi(path, preload=True )
    raw.set_montage('GSN-HydroCel-257', match_alias={'VREF' :'Cz'})
    #raw.plot_sensors(show_names=True);
    bids_path = mne_bids.BIDSPath(subject=sub,
                                session='01',
                                task= task,
                                datatype='eeg',
                                root=output_path )
    mne_bids.write_raw_bids(raw, bids_path=bids_path,  format='EEGLAB',allow_preload=True,
                            overwrite=True);

# %% Run the conversion in a loop
# List all files in the directory
files = [file for file in Path(dir_path).glob('*.mff')]

#%%
for fname in files:
    # Use regex to find the first number after "PATHS"
    match = re.search(r'(?<=PATHS_)(\d+)|(?<=PATHS_)(\d+)', fname.stem)
    if match:
        sub = match.group(1)
    convert2bids(fname, sub)
    print(f"Converted {fname} to BIDS format")

# %%

replacements = {
    "N006": "132",
    "N013": "135",
    "N014": "136"
}

for fname in files:
    for old_name, sub in replacements.items():
        if old_name in fname.name:
            print(old_name)
            convert2bids(fname, sub)
            print(f"Converted {fname} to BIDS format")
# %%
