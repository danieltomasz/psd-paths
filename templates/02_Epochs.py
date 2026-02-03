# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% tags=["parameters"]
subject_id = 127 # or whatever subject you're working with

# %%
import mne
import numpy as np
from spectral.epochs import create_epochs, get_reject_log
from spectral.utils import ProjectPaths, print_timestamp, load_config

# raw_annotated = raw_pyprep.copy()
paths = ProjectPaths(subject_id)
fname = paths.preprocessed / f"sub-{subject_id}_annotated_filtered_raw.fif"

if not fname.exists():
    raise FileNotFoundError(f"Could not find preprocessed file: {fname}")

raw_annotated = mne.io.read_raw_fif(fname, preload=True)

# %%
total_duration = raw_annotated.times[-1]

raw_annotated.compute_psd(fmin=1, fmax=50, picks="eeg", exclude="bads").plot(spatial_colors=True)

# %%


epochs = create_epochs(raw_annotated, overlap=1.5, reject_by_annotation = False)

reject_log = get_reject_log(epochs, resample=125, n_interpolate=[1], consensus=[0.40])
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

#epochs.info["bads"].extend(bad_channels_by_epochs)
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
import numpy as np

# 1. Determine if we have any bad epochs
n_bad_epochs = np.sum(reject_log.bad_epochs)
n_good_epochs = np.sum(~reject_log.bad_epochs)

# Create the figure
fig, axs = plt.subplots(2, 1, figsize=(10, 10))

# 2. Plot Bad Epochs (with Safeguard)
if n_bad_epochs > 0:
    epochs[reject_log.bad_epochs].compute_psd(
        method="welch", picks="eeg", fmax=50.0, exclude="bads"
    ).plot(
        average=False, picks="eeg", exclude="bads", show=False,
        amplitude=False, axes=axs[0]
    )
    axs[0].set_title(f"PSD for Bad Epochs (n={n_bad_epochs})")
else:
    # Handle the case where no epochs were dropped
    axs[0].text(0.5, 0.5, 'No Bad Epochs Dropped', 
                ha='center', va='center', fontsize=14)
    axs[0].set_title("PSD for Bad Epochs (None)")
    # Set dummy limits so the ylim comparison doesn't fail
    axs[0].set_ylim(0, 1) 

# 3. Plot Good Epochs (with Safeguard)
if n_good_epochs > 0:
    epochs[~reject_log.bad_epochs].compute_psd(
        method="welch", picks="eeg", fmax=50.0, exclude="bads"
    ).plot(
        average=False, picks="eeg", exclude="bads", show=False,
        amplitude=False, axes=axs[1]
    )
    axs[1].set_title(f"PSD for Good Epochs (n={n_good_epochs})")
else:
    axs[1].text(0.5, 0.5, 'No Good Epochs Remaining', 
                ha='center', va='center', fontsize=14)

# 4. Synchronize Y-Axis Limits (only if both plots have data)
if n_bad_epochs > 0 and n_good_epochs > 0:
    ylim1 = axs[0].get_ylim()
    ylim2 = axs[1].get_ylim()
    new_ylim = (min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
    axs[0].set_ylim(new_ylim)
    axs[1].set_ylim(new_ylim)

plt.tight_layout()
plt.show()

# 5. Add to Report
if 'report' in locals():
    report.add_figure(fig, title="PSD Comparison: Bad vs Good Epochs")

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

if n_bad_epochs > 0 :
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

# Do not remove bad epochs for subjects 114, 127, 128
if str(subject_id) in ['114', '127', '128']:
    print(f"⚠️  Keeping all epochs (including bad ones) for subject {subject_id}")
    epochs_good = epochs.copy()
else:
    epochs_good = epochs[~reject_log.bad_epochs]

epochs_good.save(
    f"{paths.epochs}/sub-{subject_id}_good_epochs-epo.fif", overwrite=True
)
