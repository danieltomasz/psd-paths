# ---
# title: "Preprocessing of High-Density EEG Recordings"
# format:
#   html: default
#   ipynb: default
# execute:
#   enabled: true
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 0. Load parameters
#

# %% tags=["parameters"]
# | tags: [parameters]
subject_id = 170  # or whatever subject you're working with

# %%
from pathlib import Path
import os
import mne
import matplotlib.pyplot as plt
import numpy as np

from spectral.preproc import (
    load_data,
    zapline_clean,
)

from spectral.annotation import (
    run_pyprep_cleaning,
    annotate_bads_to_raw,
    apply_potato_to_raw,
)
from spectral.utils import ProjectPaths, print_timestamp, load_config

mne.viz.set_browser_backend("matplotlib")
# mne.viz.set_browser_backend("qt")
mne.set_config("MNE_BROWSER_THEME", "light")

# Initialize paths for your subject
paths = ProjectPaths(subject_id)


# Create all directories
paths.create_directories()

# Print paths to verify
print_timestamp("Setting up project paths")
paths.show()

# %% [markdown]
# This file contains preprocessing files after downsampling them.
#

# %% [markdown]
# # 1. Preprocess raw data
#

# %%
# Load the configuration from settings toml file
# The file should be named 'settings.toml' and located in the root directory and contain the preprocessing parameters
# e.g., channels to remove, filter parameters, etc.

config = load_config()

# Access the bad channels list
bad_channels = config["preprocessing"]["channels_to_remove"]
print(f"Channels to remove: {bad_channels}")

fline = [50, 100]  # Line noise frequencies
h_freq = 40
l_freq = 1

filter_params = {
    "l_freq": l_freq,
    "h_freq":h_freq,
    "h_trans_bandwidth": "auto",
    "fir_window": "hamming",
    "fir_design": "firwin",
    "phase": "zero",
    "picks": ["ecg", "eeg"],
}

raw = load_data(subject_id, data_path=paths.data).resample(
    250, method="polyphase", verbose=True
)
total_duration = raw.times[-1]

raw_filtered = (
    raw.copy()
    .resample(250, method="polyphase", verbose=True)
    .drop_channels(bad_channels)
    .notch_filter(freqs=[fline], method="fir",   picks=['eeg', 'ecg'])
    .filter(**filter_params)
    .crop(tmin=3.0, tmax=total_duration - 3, include_tmax=True)
)

# %%
fig, ax = plt.subplots(figsize=(10, 5))
raw_psd = raw_filtered.compute_psd(
    fmax=60.0, method="welch", picks="eeg", exclude="bads"
).plot(average=False, picks="eeg", exclude="bads", show=False, axes=ax, amplitude=False)
ax.set_title(f"sub-{subject_id} PSD")

# %%
iter_param = {
    "n_iter_max": 100,
    "win_sz": 12.5,
    "spot_sz": 6,
}

filter_params = {
    "l_freq": 1,
    "h_freq": 100,
    "h_trans_bandwidth": "auto",
    "fir_window": "hamming",
    "fir_design": "firwin2",
    "phase": "zero",
    "picks": ["ecg", "eeg"],
}

raw = load_data(subject_id, data_path=paths.data)
total_duration = raw.times[-1]

raw_zap = (
    raw.copy()
    .drop_channels(bad_channels)
    .filter(**filter_params)
    .crop(tmin=3.0, tmax=total_duration - 3, include_tmax=True)
)


zap = zapline_clean(raw_zap, fline=50.0, ntimes=3, method="line", iter_param=iter_param)

# %%
filter_params = {
    "l_freq": 1,
    "h_freq": 40,
    "h_trans_bandwidth": "auto",
    "fir_window": "hamming",
    "fir_design": "firwin2",
    "phase": "zero",
    "picks": ["ecg", "eeg"],
}

zap_filtered = zap.filter(**filter_params).notch_filter(
    freqs=[50], phase="zero", method="iir"
)

fig, ax = plt.subplots(figsize=(10, 5))
raw_psd = zap_filtered.compute_psd(
    fmax=60.0, method="welch", picks="eeg", exclude="bads"
).plot(average=False, picks="eeg", exclude="bads", show=False, axes=ax, amplitude=False)
ax.set_title(f"sub-{subject_id} PSD")

# %%
#bad_channels_by_ransac = apply_pyprep(zap_filtered, output="dict", as_dict=True)

# 1. Run detection (ensure raw_filtered has a montage set)
bads_dict = run_pyprep_cleaning(raw_filtered, output_mode="all")

# 2. Annotate the raw object
# Using inplace=True prevents duplicating data in memory
raw_annotated, reasons = annotate_bads_to_raw(raw_filtered, bads_dict)

print(bad_channels_by_ransac["bad_all"])

# %%
zap_ransac = zap_filtered.copy()
zap_ransac.info["bads"].extend(bad_channels_by_ransac["bad_all"])
sensor_plot = zap_ransac.plot_sensors(show_names=True)
zap_ransac.copy().pick(bad_channels_by_ransac["bad_all"]).plot(
    duration=300.0,
    scalings=dict(eeg=1e-4),
    show_scrollbars=False,
)
zap_ransac.save(
    f"{paths.preprocessed}/sub-{subject_id}_filtered_raw.fif", overwrite=True
)

# %%
from spectral.epochs import create_epochs, get_reject_log

# raw_annotated = raw_pyprep.copy()
epochs = create_epochs(zap_ransac, overlap=1.5)

reject_log = get_reject_log(epochs, resample=125, n_interpolate=[1], consensus=[0.35])
reject_plot = reject_log.plot("vertical")


# Drop channels with more than 40% bad epochs
BADS_THRESHOLD = 0.40

bads = np.logical_or(reject_log.labels == 1, reject_log.labels == 2)

bad_channel = bads.mean(axis=0)
# print(bad_channel)
epochs.info
channel_names = epochs.info["ch_names"]
bad_channels_by_epochs = [
    name for name, value in zip(channel_names, bad_channel) if value > BADS_THRESHOLD
]
print(
    f"Additional {len(bad_channels_by_epochs)} bad channels found:{
        bad_channels_by_epochs
    }"
)

# epochs.info["bads"].extend(bad_channels_by_epochs)
try:
    bad_channel_plot_additional = epochs.plot(
        picks=bad_channels_by_epochs,
        scalings=dict(eeg=1e-4),
        show_scrollbars=False,
    )
except Exception as e:
    print(f"An error occurred: {e}")

# %%
import matplotlib.pyplot as plt

# Create a figure with two subplots
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

epochs_plot_psd = (
    epochs[reject_log.bad_epochs]
    .compute_psd(method="welch", picks="eeg", fmax=50.0, exclude="bads")
    .plot(
        average=False,
        picks="eeg",
        exclude="bads",
        show=False,
        amplitude=False,
        axes=axs[0],
    )
)
axs[0].set_title("Power Spectral Density for Bad Epochs")

epochs_plot_psd = (
    epochs[~reject_log.bad_epochs]
    .compute_psd(method="welch", picks="eeg", fmax=50.0, exclude="bads")
    .plot(
        average=False,
        picks="eeg",
        exclude="bads",
        show=False,
        amplitude=False,
        axes=axs[1],
    )
)

axs[1].set_title("Power Spectral Density for Good Epochs")


# Get the y-limits of both plots
ylim1 = axs[0].get_ylim()
ylim2 = axs[1].get_ylim()

# Set the same limits on the y-axis for both plots
axs[0].set_ylim(min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
axs[1].set_ylim(min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))


plt.show()

# %%
# Assuming 'raw' is your MNE raw object
# Get the list of all channels
all_channels = epochs.info["ch_names"]

# Get the list of bad channels
bad_channels = epochs.info["bads"]

# Get the list of good channels
good_channels = [ch for ch in all_channels if ch not in bad_channels]

# Get the indices of the good channels
good_channels_indices = [all_channels.index(ch) for ch in good_channels]

# Define the chunk size
chunk_size = 30

# Create chunks of the good channels
chunks = [
    good_channels_indices[i : i + chunk_size]
    for i in range(0, len(good_channels_indices), chunk_size)
]


# Now you can iterate over the chunks
for chunk in chunks:
    # Do something with the chunk
    # print(chunk)
    mne.viz.plot_epochs(
        epochs=epochs[reject_log.bad_epochs],
        picks=chunk,
        show=False,
        scalings=dict(eeg=1e-4),
        n_channels=chunk_size,
        show_scrollbars=False,
    )

# %%
print(f"Sub{subject_id} -  removed {sum(reject_log.bad_epochs)} epochs")
print(f"Sub{subject_id} -  retained {len(epochs[~reject_log.bad_epochs])} epochs")

epochs_good = epochs[~reject_log.bad_epochs]
epochs_good.save(
    f"{paths.epochs}/sub-{subject_id}_good_epochs-epo.fif", overwrite=True
)

# %%
from spectral.epochs import create_epochs, get_reject_log
from spectral.preproc import load_config

def find_bad_segments(raw: mne.io.Raw, 
                      epoch_length: float = 1.0, 
                      overlap: float = 0.0,
                      resample_sfreq: int = 125) -> mne.Annotations:
    """
    Identifies bad data segments using short epochs and autoreject.

    Args:
        raw (mne.io.Raw): The raw MNE data object, minimally filtered.
        epoch_length (float): The length of the short epochs for artifact detection.
        overlap (float): The overlap between short epochs.
        resample_sfreq (int): The frequency to resample to before running autoreject.

    Returns:
        mne.Annotations: MNE annotations marking the identified bad time segments.
    """
    print(f"Finding bad segments using {epoch_length}s epochs...")
    
    # 1. Create short epochs for artifact detection
    epochs = create_epochs(raw, length=epoch_length, overlap=overlap)
    
    # 2. Get the reject log from these short epochs
    reject_log = get_reject_log(epochs, resample=resample_sfreq, consensus=[0.35])
    
    # 3. Convert the reject log to annotations
    bad_annotations = reject_log_to_annotations(reject_log, epochs)
    
    print(f"Found {len(bad_annotations)} bad segments to be marked.")
    return bad_annotations

def reject_log_to_annotations(epochs: mne.Epochs, bad_epoch_indices: np.ndarray) -> mne.Annotations:
    """
    Convert a list of bad epoch indices into MNE annotations.

    Args:
        epochs: The MNE epochs object from which the reject_log was generated.
        bad_epoch_indices: NumPy array of indices for epochs to be marked as bad.

    Returns:
        mne.Annotations: Annotations object marking the bad segments.
    """
    if len(bad_epoch_indices) == 0:
        return mne.Annotations(onset=[], duration=[], description=[])

    # Get the event timings from the epochs object for the bad indices
    onsets = epochs.events[bad_epoch_indices, 0] / epochs.info["sfreq"]
    duration = len(epochs.times) / epochs.info["sfreq"]

    bad_annotations = mne.Annotations(
        onset=onsets,
        duration=[duration] * len(onsets),
        description=["bad_segment"] * len(onsets),
        orig_time=epochs.info.get('meas_date')
    )

    return bad_annotations

def find_bad_segments(raw: mne.io.Raw, 
                      epoch_length: float = 1.0, 
                      bad_epoch_threshold: float = 0.2,
                      **kwargs) -> mne.Annotations:
    """
    Identifies bad data segments using short epochs and a combined autoreject criterion.

    An epoch is marked as a "bad segment" if the fraction of its channels marked
    for rejection or interpolation exceeds `bad_epoch_threshold`.

    Args:
        raw (mne.io.Raw): The raw MNE data object, minimally filtered.
        epoch_length (float): The length of the short epochs for artifact detection.
        bad_epoch_threshold (float): Fraction of channels needing to be bad/interpolated
                                     to mark the entire epoch as a bad segment.
        **kwargs: Additional arguments passed to get_reject_log (e.g., consensus).

    Returns:
        mne.Annotations: MNE annotations marking the identified bad time segments.
    """
    print(f"Finding bad segments using {epoch_length}s epochs...")
    
    epochs = create_epochs(raw, length=epoch_length, overlap=0.0)
    reject_log = get_reject_log(epochs, **kwargs) 
    # [cite_start]Combine bad and interpolated channels to define unreliable channels [cite: 696]
    unreliable_channels = np.logical_or(reject_log.labels == 1, reject_log.labels == 2)
    
    # Calculate the fraction of unreliable channels for each epoch
    fraction_unreliable_per_epoch = unreliable_channels.mean(axis=1)
    
    # Find epochs where this fraction exceeds our threshold
    bad_epoch_indices = np.where(fraction_unreliable_per_epoch > bad_epoch_threshold)[0]
    
    # Convert the indices of these bad epochs to annotations
    bad_annotations = reject_log_to_annotations(epochs, bad_epoch_indices)
    
    print(f"Found {len(bad_annotations)} segments to annotate as 'bad'.")
    return bad_annotations


raw_for_ar = raw_filtered.copy()
epochs = create_epochs(raw_for_ar, length=3, overlap=1.5) 
reject_log = get_reject_log(epochs, resample=125, consensus=[0.35]) 

bad_annotations = find_bad_segments(
    raw_for_ar, 
    epoch_length=1.0, 
    bad_epoch_threshold=0.2,
    resample=125,
    consensus=[0.35]
)

# Add the identified bad segments as annotations
raw.set_annotations(raw.annotations + bad_annotations)
# Add these annotations to the *original* raw data
raw.set_annotations(raw.annotations + bad_annotations)
raw.save(
    f"{paths.preprocessed}/sub-{subject_id}_annotated.fif", overwrite=True
)


# %%
print(bad_annotations)
