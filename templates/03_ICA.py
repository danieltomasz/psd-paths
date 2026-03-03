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
#     display_name: psd-paths (3.13.12)
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

from spectral.viz import plot_step, plot_bad_channels, plot_epochs
from spectral.specparam import specparam2pandas
from spectral.utils import ProjectPaths,print_timestamp
from spectral.ica import compute_ica_limited, plot_annotated_topographies, plot_component_properties,get_manual_ica_list
from mne_icalabel.gui import label_ica_components
from mne_icalabel import label_components


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

# %%
epochs_good = mne.read_epochs(
    f"{paths.epochs}/sub-{subject_id}_good_epochs-epo.fif"
)

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
# Create metadata dataframe
now = datetime.now()
metadata_df = pd.DataFrame({
    "subject_id": [subject_id],
    "timestamp": [now],
    "nr_interpolated_channels": [len(epochs_ar.info["bads"])],
    "nr_dropped_ica": [len(ica.exclude)],
    "nr_retained_ica": [ica.n_components_ - len(ica.exclude)],
    "total_ica_components": [ica.n_components_],
    "nr_epochs_after_ar": [len(epochs_ar)],
    "bad_channels": [",".join(epochs_ar.info["bads"])]
})

# Save metadata to CSV
metadata_path = f"{paths.specparam}/sub-{subject_id}_ica_metadata.csv"
metadata_df.to_csv(metadata_path, index=False)
print(f"Metadata saved to: {metadata_path}")

# %%
report.save(
    f"{paths.reports}/sub-{subject_id}_report_manual_ica.html", overwrite=True
)
