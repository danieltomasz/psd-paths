# mne_pipeline.py
import mne
import matplotlib.pyplot as plt
from hamilton.function_modifiers import parameterize, tag, source, value

from spectral.utils import ProjectPaths, load_config
from spectral.preproc import load_data
from spectral.annotation import (
    run_pyprep_cleaning,
    annotate_bads_to_raw,
    apply_potato_to_raw,
)




# --- NODE: Initialize Report ---
@tag(kind="report")  # <--- Add this tag
def report_initialized(subject_id: str) -> mne.Report:
    """Creates a new MNE Report object"""
    return mne.Report(title=f"Subject {subject_id} Analysis Report", verbose=False)


# --- NODE 1: Configuration & Paths ---
def config() -> dict:
    """Loads settings.toml"""
    return load_config()


def paths(subject_id: str) -> ProjectPaths:
    """Creates project paths object"""
    paths = ProjectPaths(subject_id)
    paths.create_directories()
    return paths


def raw_loaded(subject_id: str, paths: ProjectPaths) -> mne.io.Raw:
    """Loads raw data from BIDS"""
    raw = load_data(subject_id, data_path=paths.data)
    # Handle reference channel if needed
    if "Cz" in raw.ch_names and "Cz" not in raw.info["bads"]:
        raw.info["bads"].append("Cz")
    return raw


def raw_dropped(raw_loaded: mne.io.Raw, config: dict) -> mne.io.Raw:
    """Extracts manually marked bad channels from config"""
    raw_temp = raw_loaded.copy()
    bad_channels = config["preprocessing"]["channels_to_remove"]
    return raw_temp.drop_channels(bad_channels)


def raw_potatoed(raw_dropped: mne.io.Raw) -> mne.io.Raw:
    """Applies POTATO algorithm to raw data"""
    raw_temp = raw_dropped.copy()
    raw_potato = apply_potato_to_raw(raw_temp, h_freq=40.0)
    return raw_potato


def raw_filtered(
    raw_potatoed: mne.io.Raw, filter_params: dict, fline: list
) -> mne.io.Raw:
    """Applies bandpass filter to raw data"""
    raw_temp = (
        raw_potatoed.copy()
        .resample(250, method="polyphase", verbose=True)
        .notch_filter(freqs=fline, method="fir", picks=["eeg", "ecg"])
        .filter(**filter_params)
    )

    return raw_temp


def raw_annotated_pyprep(raw_filtered: mne.io.Raw, paths: ProjectPaths) -> mne.io.Raw:
    """Runs pyprep cleaning and annotates bad segments to raw data"""
    raw_temp = raw_filtered.copy()
    annotations = run_pyprep_cleaning(raw_temp)
    raw_annotated, reasons = annotate_bads_to_raw(raw_temp, annotations)
    raw_annotated.save(
        f"{paths.preprocessed}/{paths.subject_id}_raw_annotated_filtered_raw.fif",
        overwrite=True,
    )
    return raw_annotated


# This one decorator creates TWO nodes: 'plot_raw_psd' and 'plot_clean_psd'
@tag(kind="visualization")
@parameterize(
    plot_raw_psd={
        "data_input": source("raw_potatoed"),           # Use source() for Upstream Nodes
        "stage_name": value("Raw (but potatoed)")       # Use value() for Strings/constants
    },
    plot_filtered_psd={
        "data_input": source("raw_annotated_pyprep"),  # source() links to the function logic
        "stage_name": value("Filtered (PyPrep)")        # value() passes the text directly
    }
)
def psd_plot(data_input: mne.io.Raw, subject_id: str, stage_name: str) -> plt.Figure:
    """
    Generic plotting node.
    Hamilton replaces 'data_input' with 'raw_loaded' or 'raw_potatoed' based on above.
    """
    raw_temp = data_input.copy()
    total_duration = raw_temp.times[-1]

    fig, ax = plt.subplots(figsize=(10, 5))

    raw_temp.crop(tmin=3.0, tmax=total_duration - 3, include_tmax=True)\
            .compute_psd(fmax=60.0)\
            .plot(average=False, picks="eeg", show=False, axes=ax)

    ax.set_title(f"Subject {subject_id} PSD ({stage_name})")
    plt.close(fig)
    return fig


# --- NODE: Add Plots to Report ---
@tag(kind="report")  # <--- Add this tag
def report_with_psd(
    report_initialized: mne.Report, 
    plot_raw_psd: plt.Figure,       # <--- Input 1: The Raw Plot
    plot_filtered_psd: plt.Figure   # <--- Input 2: The Filtered Plot
) -> mne.Report:
    """Adds both PSD figures to the report"""
    
    # 1. Add Raw Plot
    report_initialized.add_figure(
        fig=plot_raw_psd, 
        title="PSD (Raw Data)", 
        caption="Power Spectral Density before filtering",
        tags=("psd", "raw")
    )
    
    # 2. Add Filtered Plot
    report_initialized.add_figure(
        fig=plot_filtered_psd, 
        title="PSD (Cleaned Data)", 
        caption="Power Spectral Density after filtering and PyPrep",
        tags=("psd", "clean")
    )
    
    return report_initialized 

# --- NODE: Save Report ---
@tag(kind="report")  # <--- Add this tag
def report_saved_path(
    report_with_psd: mne.Report, 
    paths: ProjectPaths, 
    subject_id: str
) -> str:
    """
    Saves the MNE report to an HTML file.
    Returns the absolute path to the saved file.
    """
    # 1. Construct the filename
    filename = f"sub-{subject_id}_preprocessing_report.html"
    out_path = paths.reports / filename
    
    # 2. Ensure directory exists (good safety practice)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 3. Save (open_browser=False prevents it from popping up during batch runs)
    report_with_psd.save(out_path, overwrite=True, open_browser=False)
    
    return str(out_path)