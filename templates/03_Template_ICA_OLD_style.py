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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # ICA

# %% tags=["parameters"]
subject_id = "101"

# %%
import mne
from mne.preprocessing import ICA
import autoreject
from pathlib import Path
import pandas as pd
from datetime import datetime
import json
import os

from spectral.ica import compute_ica
from spectral.viz import plot_step, plot_bad_channels, plot_epochs
from spectral.specparam import specparam2pandas
from spectral.utils import ProjectPaths,print_timestamp

""" config = read_parameters()
my_paths = config_project(subject)
project_path = my_paths["project_path"]
figures_path = my_paths["figures_path"]
specparam_path = my_paths["specparam_path"]

Path(specparam_path).mkdir(parents=True, exist_ok=True)
 """


paths = ProjectPaths(subject_id)
# Create all directories
paths.create_directories()

#paths = ProjectPaths(subject)
# Create all directories
#paths.create_directories()

# This is analysis output, so it goes in the analysis folder
specparam_path = paths.specparam
project_path = paths.root
figures_path = paths.figures
preprocessed_path = paths.preprocessed
epochs_path = paths.epochs
# Print paths to verify
print_timestamp("Setting up project paths")
paths.show()

mne.viz.set_browser_backend("matplotlib")
# mne.viz.set_browser_backend("qt")
mne.set_config("MNE_BROWSER_THEME", "light")
# matplotlib.use("Agg")
#print_date_time()

# %% [markdown]
# # Start raport

# %%
clean_raw = mne.io.read_raw_fif(
    f"{paths.preprocessed}/sub-{subject_id}_annotated_filtered_raw.fif", preload=True
)

total_duration = clean_raw.times[-1]
clean_raw = clean_raw.crop(tmin=3.0, tmax=total_duration - 3, include_tmax=True)

report = mne.Report(title=f"Report for subject {subject_id}", subject=subject_id)
report.add_raw(raw=clean_raw, title="Raw", psd=True, butterfly=True, scalings='auto')

# --- HANDLES CASE WITH NO BAD CHANNELS ---
if clean_raw.info["bads"]:
    print(f"Plotting {len(clean_raw.info['bads'])} bad channels for the report.")
    
    # We use a try-except here because plotting bads can occasionally 
    # throw a ValueError if channels were marked bad but then dropped.
    try:
        bad_channel_plot = clean_raw.plot(
            picks=clean_raw.info["bads"],
            duration=min(300.0, total_duration - 6), # Ensure duration isn't longer than file
            scalings=dict(eeg=1e-4),
            show_scrollbars=False,
            show=False # Prevent popup in headless environments
        )
        report.add_figure(bad_channel_plot, title="Rejected bad channels")
    except Exception as e:
        print(f"Could not plot bad channels: {e}")
else:
    print("No bad channels detected. Skipping bad channel plot.")
    # Optional: Add a text note to the report instead of a plot
    report.add_html(
        html="<p><b>Note:</b> No bad channels were detected or marked for this subject.</p>",
        title="Rejected bad channels"
    )

# %% [markdown]
# # ICA
#
# ICA should be applied before the spatial interpolation of the
# non-functional channels.

# %%
epochs_good = mne.read_epochs(
    f"{paths.epochs}/sub-{subject_id}_good_epochs-epo.fif"
)

# %%
import mne
import numpy as np
from mne.preprocessing import ICA

# GLOBAL SETTING: Keep this as 'warn' so the rest of the pipeline 
# doesn't crash on minor floating point issues.
np.seterr(all='warn')

def compute_ica_limited(eeg_data, variance_threshold=0.99, max_cap=40, random_state=97):
    """
    Computes ICA with 99% variance, capped at max_cap.
    Safely handles Picard failure and falls back to Infomax.
    """
    # 1. Calculate the rank integer
    ranks = mne.compute_rank(eeg_data)
    actual_rank = ranks['eeg'] if 'eeg' in ranks else sum(ranks.values())
    print(f"Data rank detected: {actual_rank}")
    
    # 2. Estimate components for variance threshold using FastICA
    # Wrapped in errstate to prevent notebook-level crashes during estimation
    print(f"Estimating component count for {variance_threshold*100}% variance...")
    with np.errstate(all='ignore'):
        tmp_ica = ICA(
            n_components=variance_threshold, 
            method='fastica', 
            random_state=random_state
        )
        tmp_ica.fit(eeg_data) 
        n_variance = tmp_ica.n_components_
    
    # 3. Apply the cap
    n_final = min(n_variance, max_cap, actual_rank - 1)
    print(f"Variance threshold requested {n_variance} components.")
    print(f"Applying cap: Final ICA will use {n_final} components.")

    # 4. Attempt Picard
    try:
        print(f"Attempting Picard...")
        ica = ICA(
            n_components=n_final,
            method='picard',
            random_state=random_state,
            fit_params=dict(ortho=False, extended=True),
            max_iter=500
        )
        
        # LOCAL OVERRIDE: We raise errors ONLY here to trigger the 'except' block
        # if Picard hits a numerical stability issue.
        with np.errstate(all='raise'):
            ica.fit(eeg_data)
        print("Success: Picard converged.")

    # 5. Fallback to Infomax
    except (FloatingPointError, RuntimeWarning, ValueError, Exception) as e:
        print(f"Picard failed ({type(e).__name__}: {e}). Falling back to Infomax...")
        ica = ICA(
            n_components=n_final,
            method='infomax',
            random_state=random_state,
            fit_params=dict(extended=True),
            max_iter='auto'
        )
        # We go back to 'ignore' or 'warn' for the fallback to ensure it finishes
        with np.errstate(all='warn'):
            ica.fit(eeg_data)
        print("Success: Infomax converged.")
        
    return ica


# %%

ica = compute_ica_limited(epochs_good, variance_threshold=0.99, max_cap=40, random_state=97)
print(ica.n_components_)
ica.save(
    f"{paths.analysis}/sub-{subject_id}_my_ica_model-ica.fif", overwrite=True)


# %%
ica_plot_components = ica.plot_components()

# %%
chunk_size = 10
chunks = [
    range(i, min(i + chunk_size, ica.n_components_))
    for i in range(0, ica.n_components_, chunk_size)
]

for chunk in chunks:
    ica_plot = ica.plot_sources(
        epochs_good,
        picks=chunk,
        show_scrollbars=False,
        start=0,
        stop=len(epochs_good) - 1,
    )
    # for chunk in chunks:
    ica_plot = ica.plot_sources(
        epochs_good,
        picks=chunk,
        show_scrollbars=False,
        start=0,
        stop=20,
    )

# %%
explained_var_ratio = ica.get_explained_variance_ratio(epochs_good)
for channel_type, ratio in explained_var_ratio.items():
    print(
        f"Fraction of {
            channel_type} variance explained by all components: "
        f"{ratio}"
    )

# %% [markdown]
# # ICA label gui

# %%
from mne_icalabel.gui import label_ica_components

# gui = label_ica_components(epochs_good, ica)

# %% [markdown]
# # Removing components

# %%
ecg_indices, ecg_scores = ica.find_bads_ecg(
    clean_raw, method="correlation", threshold="auto"
)

try:
    # ica_plot = ica.plot_sources(eeg_data)
    # Exclude blink artifact components (use Fp1 as EOG proxy)

    # ica.exclude = np.argwhere(abs(eog_scores) > 0.5).ravel().tolist()
    _ = ica.plot_sources(
        epochs_good,
        show_scrollbars=False,
        picks=ecg_indices,
        start=0,
        stop=300.0,
    )
    _ = ica.plot_properties(epochs_good, picks=ecg_indices)
except Exception as e:
    # If an exception occurs, print the error message
    print("An error occurred:", e)

# %% [markdown]
# # ICA components labelling 

# %%
from mne_icalabel import label_components

def plot_annotated_topographies(ica, eeg_data, ic_labels, report=None):
    """
    Plots all ICA component topographies annotated with ICLabel names and probabilities.
    """
    labels = ic_labels["labels"]
    probs = ic_labels["y_pred_proba"]
    
    # Create titles mapping IC index to "Label (Probability%)"
    titles = {
        i: f"IC{i:02d}: {label} ({probs[i]*100:.1f}%)" 
        for i, label in enumerate(labels)
    }

    # Generate topographies. Use inst=eeg_data to ensure proper spatial context.
    figs = ica.plot_components(inst=eeg_data, title="ICA Component Classification", show=False)
    
    if not isinstance(figs, list):
        figs = [figs]

    ic_idx = 0
    for i, fig in enumerate(figs):
        # Apply custom titles to each subplot axis
        for ax in fig.axes:
            if ic_idx < len(labels):
                ax.set_title(titles[ic_idx], fontsize=9)
                ic_idx += 1
        
        # Display in notebook and add to MNE Report
        display(fig)
        if report is not None:
            report.add_figure(fig, title=f"ICA Topographies - Page {i+1}")

    return figs

def plot_component_properties(ica, eeg_data, picks, ic_labels=None, report=None, save_path=None, subject=""):
    """
    Plots detailed properties for a specific list of component indices.
    """
    if not picks:
        print("No components provided for property plotting.")
        return []

    # Get labels for titles if provided
    titles = {}
    if ic_labels:
        for i in picks:
            label = ic_labels["labels"][i]
            prob = ic_labels["y_pred_proba"][i]
            titles[i] = f"IC {i:02d}: {label} ({prob*100:.1f}%)"
    else:
        titles = {i: f"IC {i:02d}" for i in picks}

    # Generate property plots
    figs = ica.plot_properties(eeg_data, picks=picks, show=False, psd_args={"fmax": 100.0})

    for i, fig in zip(picks, figs):
        fig.suptitle(titles[i], fontsize=12)
        display(fig)
        
        # Save to disk if path is provided
        if save_path:
            label_slug = titles[i].split(":")[1].split("(")[0].strip().replace(" ", "-")
            filename = f"sub-{subject}_IC{i:02d}_{label_slug}.png"
            fig.savefig(f"{save_path}/{filename}", dpi=300, bbox_inches="tight")

        # Add to MNE Report
        if report is not None:
            report.add_figure(fig, title=f"Properties {titles[i]}", tags=("ica", "artifact"))

    return figs

def get_manual_ica_list(subject_id,  log_path):
    """
    Checks the CSV for manual overrides. 
    Returns a list of indices if found, else returns an empty list.
    """
    if os.path.exists(log_path):
        df_log = pd.read_csv(log_path)
        # Match subject_id as a string
        sub_row = df_log[df_log['subject_id'].astype(str) == str(subject_id)]
        
        if not sub_row.empty:
            manual_val = sub_row.iloc[0]['manual_exclude']
            
            # --- ROBUST CHECK FOR NAN AND EMPTY STRINGS ---
            if pd.isna(manual_val) or str(manual_val).strip().lower() == 'nan' or str(manual_val).strip() == '':
                return [] # Fall back to automated detection
            # ----------------------------------------------

            print(f"✅ Using manual override from CSV for sub-{subject_id}")
            # Convert "0, 1, 2" string to [0, 1, 2]
            return [int(x.strip()) for x in str(manual_val).split(',') if x.strip().isdigit()]
            
    return []



# %%
# 1. Classify
ic_labels = label_components(clean_raw, ica, method="iclabel")

# 2. Plot all topographies with their labels
plot_annotated_topographies(ica, clean_raw, ic_labels, report=report)

# %%
all_ics = list(range(ica.n_components_))
# 2. Call your function with the full list
plot_component_properties(
    ica, 
    clean_raw, 
    picks=all_ics,        # This passes every component index
    ic_labels=ic_labels, 
    report=report,
    save_path=figures_path,
    subject=subject_id
)

# %% [markdown]
# # Select component to removal 

# %%


# 3. Identify components to exclude (Manual ECG + Automated Artifacts)
artifacts = ["eye blink", "muscle artifact", "heart beat", "line noise", "channel noise"]
bad_indices = [
    i for i, label in enumerate(ic_labels["labels"]) 
    if label in artifacts and ic_labels["y_pred_proba"][i] > 0.80
]

# Merge with your manual ecg_indices from the previous step
final_picks = list(set(bad_indices + list(ecg_indices)))

# 4. Plot properties for ONLY the components we are worried about
plot_component_properties(
    ica, 
    clean_raw, 
    picks=final_picks, 
    ic_labels=ic_labels, 
    report=report,
    save_path=figures_path,
    subject=subject_id
)

# 5. Set the exclusion list
ica.exclude = final_picks

# %%
# --- NEW: Plot Time Series (Sources) ---
if final_picks:
    print(f"Plotting time series for components: {final_picks}")
    # We plot a 20-second window by default to see the artifact patterns clearly
    fig_sources = ica.plot_sources(
        clean_raw, 
        picks=final_picks, 
        start=0, 
        stop=20, 
        show_scrollbars=False,
        show=True
    )
    
    if report is not None:
        report.add_figure(fig_sources, title="Excluded Components Time Series (Sources)")

# %%
import pandas as pd
import os

# Define the central log path
log_path = f"{paths.outputs}/ica_cleaning_master_log.csv"



# 2. Check for manual override in the CSV
manual_picks = get_manual_ica_list(subject_id, log_path)
print(manual_picks)

# 3. Final Decision Logic:
# If manual is non-empty, replace the list; else use the automated list
if manual_picks:
    ica.exclude = manual_picks
    print(f"Final exclusion (Manual): {ica.exclude}")
else:
    ica.exclude = final_picks
    print(f"Final exclusion (Automated): {ica.exclude}")

final_exclude = ica.exclude
# C. Apply and Save
epochs_clean = ica.apply(epochs_good.copy(), exclude=ica.exclude)

epochs_clean.save(f"{paths.analysis}/sub-{subject_id}_clean_ica-manual-epo.fif", overwrite=True)

# D. Update the Master CSV Log
new_entry = {
    "subject_id": str(subject_id), # Force current ID to string
    "auto_detected": ",".join(map(str, final_picks)),
    "manual_exclude": "", 
    "bad_channels": ",".join(clean_raw.info["bads"]),
    "status": "Auto-Cleaned"
}

if not os.path.exists(log_path):
    df_log = pd.DataFrame([new_entry])
else:
    df_log = pd.read_csv(log_path)
    
    # FIX: Convert existing subject_id column to strings to prevent mixed-type errors
    df_log['subject_id'] = df_log['subject_id'].astype(str)
    
    # Update existing subject or append new
    df_log = df_log[df_log['subject_id'] != str(subject_id)] # Remove old entry
    df_log = pd.concat([df_log, pd.DataFrame([new_entry])], ignore_index=True)

# Now sort_values will work because all IDs are strings
df_log.sort_values("subject_id").to_csv(log_path, index=False)
print(f"📝 Master log updated at: {log_path}")

# %%
ica.exclude

# %%
# Add location of bad channels
try:
    report.add_ica(
        ica=ica,
        title="ICA cleaning",
        picks=ica.exclude,  # plot the excluded EOG components
        inst=epochs_good,
    )
    # Add just sources of the removed components
    ica_figure_sources = ica.plot_sources(
        epochs_good, picks=final_exclude, show_scrollbars=False
    )
    report.add_figure(ica_figure_sources, title="ICA rejected components")

except Exception as e:
    # If an exception occurs, print the error message
    print("An error occurred:", e)

# %% [markdown]
# # ICA plot overlay

# %%
cleaned_plot = ica.plot_overlay(epochs_good.average())

# %%
epochs_plot, epochs_timeseries, epochs_plot_psd = plot_epochs(
epochs_clean, figures_path=figures_path, subject=subject_id
)
report.add_figure(epochs_plot, title="Cleaned epochs after ICA")
report.add_figure(epochs_plot_psd, title="Cleaned epochs after ICA - PSD")

# %%
ica.save(
    f"{paths.analysis}/sub-{subject_id}_my_ica_model-ica.fif", overwrite=True)

# %% [markdown]
# # Final part of the analysis

# %%
ica_loaded = mne.preprocessing.read_ica(
    f"{paths.analysis}/sub-{subject_id}_my_ica_model-ica.fif"
)

# %%
auto_reject_pre_ica = autoreject.AutoReject(
    n_interpolate=[1, 2, 4, 8, 32, 64],
    n_jobs=-1,
    random_state=100,
    thresh_method="bayesian_optimization",
    verbose=False,
    # n_interpolate=np.array([0]),
    # consensus=0.8,
).fit(epochs_clean)
print("fitting finished")
epochs_ar, reject_log = auto_reject_pre_ica.transform(
   epochs_clean, return_log=True
)


# depending of the resampling - different number of epochs are rejected

reject_plot = reject_log.plot("vertical")
report.add_figure(
    reject_plot, title="Interpolated epochs - AutoReject overview")

# Add the last part to the report
report.add_figure(
    reject_plot, title="Autoreject - Rejected and interpolated epochs after ICA"
)

# %%
epochs_interpolated = epochs_ar.copy().interpolate_bads(exclude=["VREF"])
epochs_interpolated.save(
    f"{paths.analysis}/sub-{subject_id}_interpolated-epo.fif", overwrite=True
)

# %%
# epochs_interpolated
epochs_plot, epochs_timeseries, epochs_plot_psd = plot_epochs(
    epochs_interpolated, figures_path, subject_id, stage="final_interpolared_epochs"
)
# Add figures to the report
report.add_figure(epochs_plot, title="Interpolated epochs")
report.add_figure(epochs_plot_psd, title="Interpolated epochs - PSD")

# %%
from specparam.plts.spectra import plot_spectra
from specparam import SpectralGroupModel


fg = SpectralGroupModel(
    peak_width_limits=[1, 6],
    min_peak_height=0.15,
    peak_threshold=2.0,
    max_n_peaks=6,
    verbose=False,
)

freq_range = [2, 35]

# %%
psd = epochs_interpolated.compute_psd().average()
spectra, freqs = psd.get_data(return_freqs=True)
# Initialize a FOOOFGroup object, with desired settings

# Define the frequency range to fit

with np.errstate(divide='ignore', invalid='ignore', over='ignore'):
    fg.fit(freqs, spectra, freq_range)
fg.plot()

# %%
channel_names = epochs_interpolated.info["ch_names"]
df_channels = pd.DataFrame({"ID": range(len(channel_names)), "ch": channel_names})

df = specparam2pandas(fg)
df = df.merge(df_channels, on="ID")
df["sub_id"] = subject_id


# Get the current date and time
now = datetime.now()
df["timestamp"] = now
df["nr_intepolated_channels"] = len(epochs_ar.info["bads"])
df["nr_dropped_ica"] = len(ica.exclude)
df["nr_retained_ica"] = ica.n_components_ - len(ica.exclude)
# Create a new list of column names
cols = ["ch"] + [col for col in df.columns if col != "ch"]

# Reorder the columns
df = df[cols]

# %% [markdown]
# # Plot R and exponent across the scalp

# %%
fg.to_df(None)

# %%
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import mne  # Make sure you have the MNE library installe

# Extract aperiodic exponent values
# Extract aperiodic exponent values (Updated)
# For the exponent (aperiodic parameters)
results_df = fg.to_df()

# 2. Extract the values directly from the DataFrame columns
# In 2.0rc6, these columns are named 'exponent' and 'r2'
exps = results_df['exponent'].values
r_squared = results_df['gof_rsquared'].values
# Assuming 'exps' is your data array and 'raw' is an MNE raw object
# Also, assuming 'unit_label' and 'fontsize' variables are defined

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

# The 'cmap' parameter expects a colormap object, not a string
im1, _ = mne.viz.plot_topomap(
    exps, epochs_interpolated.info, axes=axs[0], cmap="viridis", contours=0, show=False
)
axs[0].set_title("Exponent Values")

#  Colorbar setup for the first subplot at the bottom
cbar_ax1 = fig.add_axes([0.1, 0.05, 0.35, 0.03])
fig.colorbar(im1, cax=cbar_ax1, orientation="horizontal")


# Plot the 'errors' data in the second subplot
im2, _ = mne.viz.plot_topomap(
    r_squared,
    epochs_interpolated.info,
    axes=axs[1],
    cmap="plasma",
    contours=0,
    show=False,
)
axs[1].set_title("R_squared Values")

# Colorbar setup for the second subplot at the bottom
cbar_ax2 = fig.add_axes([0.55, 0.05, 0.35, 0.03])
fig.colorbar(im2, cax=cbar_ax2, orientation="horizontal")
fig.suptitle(f"sub-{subject_id} - Exponent and R_squared values")

plt.show()
report.add_figure(fig, title="Exponent and R_squared values")

# %%
# Compare the power spectra between low and high exponent channels
fig, ax = plt.subplots(1, 2, figsize=(12, 6))


def argmedian(arr):
    return np.argsort(arr)[len(arr) // 2]


# Updated attribute: .power_spectrum -> .spectrum
spectra_exp = [
    fg.get_model(np.argmin(exps)).data.power_spectrum,
    fg.get_model(argmedian(exps)).data.power_spectrum,
    fg.get_model(np.argmax(exps)).data.power_spectrum,
]


labels_spectra_exp = [
    f"Low Exponent {format(np.min(exps), '.2f')}",
    f"Median Exponent {format(np.median(exps), '.2f')}",
    f"High Exponent {format(np.max(exps), '.2f')}",
]

plot_spectra(
    fg.data.freqs,
    spectra_exp,
    ax=ax[0],
    labels=labels_spectra_exp,
)
# Do the same for the R-squared spectra
spectra_r_squared = [
    fg.get_model(np.argmin(r_squared)).data.power_spectrum,
    fg.get_model(argmedian(r_squared)).data.power_spectrum,
    fg.get_model(np.argmax(r_squared)).data.power_spectrum,
]

labels_spectra_r_squared = [
    f"Low R_squared  {format(np.min(r_squared), '.2f')}",
    f"Median R_squared {format(np.median(r_squared), '.2f')}",
    f"High R_squared {format(np.max(r_squared), '.2f')}",
]


my_colors = ["blue", "green", "red"]
plot_spectra(
    fg.data.freqs,
    spectra_r_squared,
    ax=ax[1],
    labels=labels_spectra_r_squared,
    colors=my_colors,
)
ylim1 = ax[0].get_ylim()
ylim2 = ax[1].get_ylim()
# Set the same limits on the y-axis for both plots
ax[0].set_ylim(min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
ax[1].set_ylim(min(ylim1[0], ylim2[0]), max(ylim1[1], ylim2[1]))
fig.suptitle(
    f"sub-{subject_id} - Power spectra comparison between low, median and high exponent and R_squared values"
)

report.add_figure(
    fig, title="Examples of spectra as a function of exponent and R_squared values"
)

# %%
spectra_exp_fm = [
    fg.get_model(np.argmin(exps), regenerate=True),
    fg.get_model(argmedian(exps), regenerate=True),
    fg.get_model(np.argmax(exps), regenerate=True),
]
for fm, label in zip(spectra_exp_fm, labels_spectra_exp):
    # fm = fg.get_fooof(ind=2, regenerate=True)
    # Print results and plot extracted model fit
    fm.print_results()
    fm.plot()
    print(label)

# %%
spectra_r_squared_fm = [
    fg.get_model(np.argmin(r_squared), regenerate=True),
    fg.get_model(argmedian(r_squared), regenerate=True),
    fg.get_model(np.argmax(r_squared), regenerate=True),
]

for fm, label in zip(spectra_r_squared_fm, labels_spectra_r_squared):
    # fm = fg.get_fooof(ind=2, regenerate=True)
    # Print results and plot extracted model fit
    fm.print_results()
    fm.plot()
    print(label)

# %%
df.to_csv(f"{paths.specparam}/sub-{subject_id}-specparam.csv", index=False)
print(f"Subject {subject_id} done")

# %%
report.save(
    f"{paths.reports}/sub-{subject_id}_report_manual.html", overwrite=True
)
