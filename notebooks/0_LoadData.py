# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: psd-paths
#     language: python
#     name: python3
# ---

# %%
"""
Standalone MFF to BIDS converter.
No configuration files required.
"""
from pathlib import Path
import re
import mne
from mne_bids import BIDSPath, write_raw_bids

# ==========================================
# 1. INPUT & OUTPUT PATHS (EDIT HERE)
# ==========================================
# Input: The folder where your .mff files are located
raw_dir_path = "/Users/daniel/PhD/Projects/psd-paths/data/orginal"

# Output: The folder where you want the BIDS structure to be created
output_dir = "/Users/daniel/PhD/Projects/psd-paths/data/bids"
# ==========================================


def convert2bids(
    file_path, sub_id, task_name="RESTING", bids_root=output_dir, montage=None
):
    """
    Reads MFF and writes to BIDS (BrainVision format)
    without generating estimated coordinate files.
    """
    try:
        # Read the raw data (preload=False prevents RAM crashes)
        raw = mne.io.read_raw_egi(file_path, preload=False, verbose=False)

        # Rename VREF to Cz (Standard cleanup for EGI nets)

        if montage is not None:
            raw.set_montage(montage, match_alias={"VREF": "Cz"})
        elif "VREF" in raw.ch_names:
            raw.rename_channels({"VREF": "Cz"})
        # Define the BIDS target path
        bids_path = BIDSPath(
            subject=sub_id, session="01", task=task_name, datatype="eeg", root=bids_root
        )

        # Write to BIDS
        # defaults to BrainVision format (.vhdr) which is best for BIDS
        write_raw_bids(raw, bids_path=bids_path, overwrite=True, verbose=False)

        print(f"✔ Success: sub-{sub_id}")

    except Exception as e:
        print(f"❌ Failed: sub-{sub_id} | Error: {e}")


# ==========================================
# 2. MAIN EXECUTION LOOP
# ==========================================

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Find all .mff files in the input directory
mff_files = list(Path(raw_dir_path).glob("*.mff"))
print(f"Found {len(mff_files)} MFF files. Starting conversion...\n")

for file_path in mff_files:
    # --- Logic to extract Subject ID ---
    # Looks for 'sub-' followed by digits in the filename (e.g., sub-128...)
    match = re.search(r"sub-(\d+)", file_path.stem, re.IGNORECASE)

    if match:
        sub_id = match.group(1)

        # --- Logic to determine Task ---
        # Checks if 'REST' is in the filename, otherwise defaults to 'task'
        task = "RESTING" if "RESTING" in file_path.stem.upper() else "RESTING"

        # Run conversion
        convert2bids(file_path, sub_id, task_name=task, montage="GSN-HydroCel-257")

    else:
        print(f"⚠️ Skipping {file_path.name}: Could not find 'sub-xxx' pattern.")
