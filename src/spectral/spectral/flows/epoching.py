# mne_pipeline.py
import mne
import numpy as np
import matplotlib.pyplot as plt
from hamilton.function_modifiers import parameterize, tag, source, value
from spectral.utils import ProjectPaths, load_config
from spectral.epochs import create_epochs, get_reject_log, plot_rejection_matrix
from spectral.helpers import load_or_create_report

@tag(kind="report")
def report_loaded(
    paths: ProjectPaths, 
    subject_id: str,
    append_to_existing: bool = False  # Start fresh for preprocessing
) -> mne.Report:
    """Loads or creates report for preprocessing"""
    return load_or_create_report(
        paths, 
        subject_id, 
        append_to_existing,
        stage="Preprocessing"
    )

# --- NODE 1: Configuration & Paths ---
def config() -> dict:
    """Loads settings.toml"""
    return load_config()


def paths(subject_id: str) -> ProjectPaths:
    """Creates project paths object"""
    paths = ProjectPaths(subject_id)
    paths.create_directories()
    return paths


# --- NODE: Load Preprocessed Data (if starting fresh from disk) ---
def raw_annotated_loaded(paths: ProjectPaths, subject_id: str) -> mne.io.Raw:
    """
    Loads the preprocessed/annotated .fif file saved after PyPrep.
    Adjust the filename pattern to match what you saved earlier.
    """
    # Based on your notebook: f"{paths.preprocessed}/sub-{subject_id}_filtered_raw.fif"
    # Or f"{paths.preprocessed}/sub-{subject_id}_annotated.fif"
    fname = paths.preprocessed / f"sub-{subject_id}_annotated_filtered_raw.fif"

    if not fname.exists():
        raise FileNotFoundError(f"Could not find preprocessed file: {fname}")

    return mne.io.read_raw_fif(fname, preload=True)


# --- NODE: Create Epochs ---
def epochs_all(
    raw_annotated_loaded: mne.io.Raw, duration: float = 3.0, overlap: float = 1.5
) -> mne.Epochs:
    """
    Cuts the raw data into fixed-length epochs (sliding window).
    """
    # Uses your wrapper from spectral.epochs
    return create_epochs(raw_annotated_loaded, length=duration, overlap=overlap)


# --- NODE: Calculate Rejection Log (Autoreject) ---
def rejection_log(
    epochs_all: mne.Epochs,
    n_interpolate: list = [1],
    consensus: list = [0.35],
    resample_freq: int = 125,
) -> object:
    """
    Runs Autoreject to identify bad epochs.
    Returns the reject_log object.
    """
    copy
    # Uses your wrapper from spectral.epochs
    return get_reject_log(
        epochs_all,
        resample=resample_freq,
        n_interpolate=n_interpolate,
        consensus=consensus,
    )


# --- NODE: Apply Rejection (Get Clean Epochs) ---
def epochs_clean(epochs_all: mne.Epochs, rejection_log: object) -> mne.Epochs:
    """
    Drops the bad epochs identified by the rejection log.
    """
    # Invert the bad_epochs boolean array to get Good epochs
    return epochs_all[~rejection_log.bad_epochs]


# --- NODE: Save Clean Epochs ---
def saved_epochs_path(
    epochs_clean: mne.Epochs, paths: ProjectPaths, subject_id: str
) -> str:
    """
    Saves the cleaned epochs to disk.
    """
    out_path = paths.epochs / f"sub-{subject_id}_good_epochs-epo.fif"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    epochs_clean.save(out_path, overwrite=True)
    return str(out_path)


@tag(kind="visualization")
def plot_epoch_rejection(
    epochs_all: mne.Epochs, rejection_log: object, subject_id: str
) -> plt.Figure:
    """
    Plots PSD comparison between Good and Bad epochs.
    """
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))

    # 1. Plot Bad Epochs
    bad_epochs = epochs_all[rejection_log.bad_epochs]
    if len(bad_epochs) > 0:
        bad_epochs.compute_psd(fmax=50.0).plot(
            average=False, picks="eeg", show=False, axes=axs[0], amplitude=False
        )
        axs[0].set_title(f"Bad Epochs (n={len(bad_epochs)})")
    else:
        axs[0].text(0.5, 0.5, "No Bad Epochs Found", ha="center")

    # 2. Plot Good Epochs
    good_epochs = epochs_all[~rejection_log.bad_epochs]
    if len(good_epochs) > 0:
        good_epochs.compute_psd(fmax=50.0).plot(
            average=False, picks="eeg", show=False, axes=axs[1], amplitude=False
        )
        axs[1].set_title(f"Good Epochs (n={len(good_epochs)})")

    # Match Y-axis limits for fair comparison
    try:
        ylim1 = axs[0].get_ylim()
        ylim2 = axs[1].get_ylim()
        common_min = min(ylim1[0], ylim2[0])
        common_max = max(ylim1[1], ylim2[1])
        axs[0].set_ylim(common_min, common_max)
        axs[1].set_ylim(common_min, common_max)
    except:
        pass  # Handle cases where plot might be empty

    fig.suptitle(f"Subject {subject_id} Epoch Rejection")
    plt.close(fig)
    return fig


@tag(kind="visualization")
def plot_rejection_log(rejection_log: object, subject_id: str) -> plt.Figure:
    """Visualizes the rejection log as a matrix"""
    fig = plot_rejection_matrix(rejection_log=rejection_log, subject=subject_id)
    plt.close(fig)
    return fig


@tag(kind="report")
def report_with_epochs(
    report_loaded: mne.Report,  # Changed from report_with_psd
    plot_epoch_rejection: plt.Figure,
    plot_rejection_log: plt.Figure,
) -> mne.Report:
    """Adds the Epoch analysis plots to the existing report"""
    report_loaded.add_figure(
        fig=plot_rejection_log,
        title="Epoch Cleaning Analysis",
        caption="Comparison of Power Spectral Density between rejected and retained epochs.",
        tags=("epochs", "cleaning"),
    )
    report_loaded.add_figure(
        fig=plot_epoch_rejection,
        title="Epoch Cleaning Analysis",
        caption="Comparison of Power Spectral Density between rejected and retained epochs.",
        tags=("epochs", "cleaning"),
    )

    return report_loaded


@tag(kind="report")
def report_epochs_saved_path(
    report_with_epochs: mne.Report, paths: ProjectPaths, subject_id: str
) -> str:
    """Saves the updated report with epoch analysis."""
    base_name = f"sub-{subject_id}_report"
    h5_path = paths.reports / f"{base_name}.h5"
    html_path = paths.reports / f"{base_name}.html"

    report_with_epochs.save(h5_path, overwrite=True)
    report_with_epochs.save(html_path, overwrite=True, open_browser=False)

    return str(html_path)

