"""Functions to plot data and PSD"""

from pathlib import Path
import matplotlib.pyplot as plt
import mne
import numpy as np


def plot_step(
    temp, subject, figures_path, stage="raw", duration=50.0, n_channels=50, fmax=100.0
):
    """Plot raw data and PSD of the data"""
    raw_plot = mne.viz.plot_raw(
        temp.copy(),
        duration=duration,
        scalings=dict(eeg=1e-4),
        n_channels=n_channels,
        show_scrollbars=False,
        title=f"sub-{subject}_{stage}",
    )
    Path(figures_path).mkdir(parents=True, exist_ok=True)
    raw_plot.savefig(
        f"{figures_path}/sub-{subject}_{stage}.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot is saved to monitor the quality of the data
    fig, ax = plt.subplots(figsize=(10, 5))
    raw_psd = temp.compute_psd(
        fmax=fmax, method="welch", picks="eeg", exclude="bads"
    ).plot(
        average=False, picks="eeg", exclude="bads", show=False, axes=ax, amplitude=False
    )
    ax.set_title(f"sub-{subject} {stage} PSD")
    raw_psd.savefig(
        f"{figures_path}/sub-{subject}_{stage}-psd.png", dpi=300, bbox_inches="tight"
    )
    fig.show()


def plot_epochs(
    epochs,
    figures_path,
    subject,
    stage="epochs",
    n_epochs=10,
    n_channels=10,
    fmax=100.0,
):
    """Plot the  epoch data"""
    epochs_plot = epochs.copy().average().detrend().plot_joint()
    Path(figures_path).mkdir(parents=True, exist_ok=True)
    epochs_plot.savefig(
        f"{figures_path}/sub-{subject}_{stage}.png", dpi=300, bbox_inches="tight"
    )
    epochs_timeseries = mne.viz.plot_epochs(
        epochs=epochs,
        picks="eeg",
        show=False,
        n_epochs=n_epochs,
        n_channels=n_channels,
        scalings=dict(eeg=1e-4),
        show_scrollbars=False,
    )
    epochs_plot_psd = epochs.compute_psd(
        method="welch", picks="eeg", fmax=fmax, exclude="bads"
    ).plot(average=False, picks="eeg", exclude="bads", show=False, amplitude=False)
    epochs_plot_psd.savefig(
        f"{figures_path}/sub-{subject}_{stage}_psd.png", dpi=300, bbox_inches="tight"
    )
    return (epochs_plot, epochs_timeseries, epochs_plot_psd)


def plot_bad_channels(raw, subject, figures_path):
    """Plot the sensor locations"""
    Path(figures_path).mkdir(parents=True, exist_ok=True)
    bad_channels = raw.copy().pick(raw.info["bads"])
    bad_channel_plot = bad_channels.plot(
        duration=300.0,
        scalings=dict(eeg=1e-4),
        show_scrollbars=False,
    )
    # Path(figure_path).mkdir(parents=True, exist_ok=True)
    bad_channel_plot.savefig(
        f"{figures_path}/sub-{subject}_bad_channel_plot.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    sensor_plot = raw.plot_sensors(show_names=True)
    sensor_plot.savefig(
        f"{figures_path}/sub-{subject}_sensors.png", dpi=300, bbox_inches="tight"
    )
    plt.close()



