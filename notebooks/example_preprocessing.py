import marimo

__generated_with = "0.17.6"
app = marimo.App(width="medium", sql_output="polars")


@app.cell
def _():
    import marimo as mo
    return


@app.cell
def _():
    from pathlib import Path
    import os
    import mne
    import matplotlib.pyplot as plt
    import numpy as np

    from spectral.preproc import (
        load_data,
        zapline_clean,
        apply_pyprep,
    )
    from spectral.utils import ProjectPaths, print_timestamp, load_config

    mne.viz.set_browser_backend("matplotlib")
    # mne.viz.set_browser_backend("qt")
    mne.set_config("MNE_BROWSER_THEME", "light")
    return ProjectPaths, load_config, load_data, plt, print_timestamp


@app.cell
def _(ProjectPaths, print_timestamp):
    subject_id = 101  # or whatever subject you're working with

    # Initialize paths for your subject
    paths = ProjectPaths(subject_id)

    # Create all directories
    paths.create_directories()

    # Print paths to verify
    print_timestamp("Setting up project paths")
    paths.show()
    return paths, subject_id


@app.cell
def _(load_config, load_data, paths, subject_id):
    from mne_denoise import apply_zapline_to_raw

    # https://github.com/mne-tools/mne-denoise
    config = load_config()
    bad_channels = config["preprocessing"]["channels_to_remove"]
    print(f"Channels to remove: {bad_channels}")

    raw = load_data(subject_id, data_path=paths.data).resample(
        250, method="polyphase", verbose=True
    )
    raw_clean, *_ = apply_zapline_to_raw(raw, line_freqs="line", plotResults=False)

    # Plot and make it interactive
    # Capture the figure returned by plot()
    fig1 = raw_clean.plot(show=False, block=False)
    fig1
    return bad_channels, raw, raw_clean


@app.cell
def _(raw_clean):
    # Use PSD plot which works better in notebooks
    fig2 = raw_clean.compute_psd().plot(show=False)
    fig2
    return


@app.cell
def _(bad_channels, raw):
    fline = [50, 100]  # Line noise frequencies
    h_freq = 40
    l_freq = 1

    filter_params = {
        "l_freq": l_freq,
        "h_freq": h_freq,
        "h_trans_bandwidth": "auto",
        "fir_window": "hamming",
        "fir_design": "firwin",
        "phase": "zero",
        "picks": ["ecg", "eeg"],
    }


    total_duration = raw.times[-1]

    raw_filtered = (
        raw.copy()
        .resample(250, method="polyphase", verbose=True)
        .drop_channels(bad_channels)
        .notch_filter(freqs=[fline], method="fir", picks=["eeg", "ecg"])
        .filter(**filter_params)
        .crop(tmin=3.0, tmax=total_duration - 3, include_tmax=True)
    )
    return (raw_filtered,)


@app.cell
def _(plt, raw_filtered, subject_id):
    fig, ax = plt.subplots(figsize=(10, 5))
    raw_psd = raw_filtered.compute_psd(
        fmax=60.0, method="welch", picks="eeg", exclude="bads"
    ).plot(
        average=False,
        picks="eeg",
        exclude="bads",
        show=False,
        axes=ax,
        amplitude=False,
    )
    ax.set_title(f"sub-{subject_id} PSD")
    return


if __name__ == "__main__":
    app.run()
