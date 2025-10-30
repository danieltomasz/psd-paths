import marimo

__generated_with = "0.17.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo 
    return


@app.cell
def _():
    import mne
    from mne.preprocessing import ICA
    import autoreject
    import pandas as pd
    from datetime import datetime
    import json
    from pathlib import Path
    from spectral.ica import compute_ica
    from spectral.viz import plot_epochs
    return (mne,)


@app.cell
def _():
    # Import model objects, and Bands object to define bands of interest
    from specparam import SpectralModel, SpectralGroupModel, Bands

    # Import simulation functions to create some example data
    from specparam.sim import sim_power_spectrum, sim_group_power_spectra
    # Initialize bands object, defining alpha band
    bands1 = Bands({'alpha' : [7, 14]})

    # Initialize model object
    fm = SpectralModel()

    # Simulate example power spectrum
    freqs, powers = sim_power_spectrum(\
        [1, 50], {'knee' : [0, 10, 1]}, {'gaussian' : [10, 0.25, 2]}, freq_res=0.25)

    # Fit model to power spectrum
    fm.fit(freqs, powers)
    fm.to_df(None)
    return SpectralGroupModel, sim_group_power_spectra


@app.cell
def _(SpectralGroupModel, sim_group_power_spectra):
    # Simulate an example group of power spectra
    freqs, powers = sim_group_power_spectra(\
        5, [1, 50], {'fixed' : [0, 1]}, {'gaussian' : [10, 0.25, 2]})

    # Initialize a group model object and fit power spectra
    fg = SpectralGroupModel(verbose=False)
    fg.fit(freqs, powers)
    return


@app.cell
def _(mne):

    from spectral.specparam import specparam2pandas


    from spectral.utils import ProjectPaths,print_timestamp

    mne.viz.set_browser_backend("matplotlib")
    # mne.viz.set_browser_backend("qt")
    mne.set_config("MNE_BROWSER_THEME", "light")
    # matplotlib.use("Agg")
    #print_date_time()
    subject = 101


    paths = ProjectPaths(subject)
    # Create all directories
    paths.create_directories()

    # This is analysis output, so it goes in the analysis folder
    specparam_path = paths.specparam
    project_path = paths.root
    figures_path = paths.figures
    preprocessed_path = paths.preprocessed
    epochs_path = paths.epochs
    # Print paths to verify
    print_timestamp("Setting up project paths")
    paths.show()
    return


if __name__ == "__main__":
    app.run()
