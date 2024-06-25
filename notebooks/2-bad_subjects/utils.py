import os
import copy
import pandas as pd
import numpy as np
import mne
import matplotlib.pyplot as plt
from mne.preprocessing import ICA
from mne_icalabel.gui import label_ica_components
import autoreject
from specparam.plts.spectra import plot_spectra
from specparam import SpectralGroupModel
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import numpy as np
import re
import pandas as pd
import numpy as np
from specparam.core.funcs import infer_ap_func
from specparam.core.info import get_ap_indices
import datetime


def compare_before_after(
    epochs_before,
    epochs_after,
    subject: int,
    fmax: float = 40.0,
    title: str | None = None,
):
    """Compare PSD of the data befre and after cleaning"""
    if title is None:
        title = f"ICA Comparison for subject {subject}"
    fig_psd, axs = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(10, 8))

    epochs_before.compute_psd(
        method="welch", picks="eeg", fmax=fmax, exclude="bads"
    ).plot(
        average=False,
        picks="eeg",
        exclude="bads",
        show=False,
        amplitude=False,
        axes=axs[0],
    )
    axs[0].set_title(f"Before  for subject {subject}")

    epochs_after.compute_psd(
        method="welch", picks="eeg", fmax=fmax, exclude="bads"
    ).plot(
        average=False,
        picks="eeg",
        exclude="bads",
        show=False,
        amplitude=False,
        axes=axs[1],
    )
    axs[1].set_title(f"After for subject {subject}")
    fig_psd.suptitle(title, fontsize=16)

    return fig_psd


def create_epochs(
    raw: mne.io.Raw,
    epochs_params: dict | None = None,
    length: float = 5,
    overlap: float = 1.5,
):
    """Create epochs from raw data"""
    # Create fixed length events
    events = mne.make_fixed_length_events(
        raw=raw, start=0, stop=None, duration=length, overlap=overlap, first_samp=True
    )

    # Default parameters for epochs
    default_params = {
        "raw": raw,
        "events": events,
        "tmin": 0,
        "tmax": length,
        "detrend": 1,  # Linear detrending
        "baseline": None,
        "preload": True,
        "reject_by_annotation": None,
        "reject": None,
    }
    # Update default parameters with the ones passed to the function
    if epochs_params is not None:
        default_params.update(epochs_params)

    return mne.Epochs(**default_params)


def switch_bad_to_interpolate(reject_log):
    """
    Switch all bad epochs to epochs to be interpolated in a RejectLog instance.

    Parameters
    ----------
    reject_log : RejectLog
        The RejectLog instance to modify.

    Returns
    -------
    RejectLog
        A new RejectLog instance with the modifications applied.
    """

    # Create a copy of the original labels
    new_labels = reject_log.labels.copy()

    # Find all bad epochs
    bad_epoch_indices = np.where(reject_log.bad_epochs)[0]

    # Print the indexes of bad epochs
    print("Indexes of bad epochs:")
    print(bad_epoch_indices)  # epochs doesnt start from 0

    # For each bad epoch, set all channels to be interpolated (value 2)
    for epoch_idx in bad_epoch_indices:
        new_labels[epoch_idx, :] = 2

    # Create a new RejectLog instance with the modified labels
    new_reject_log = autoreject.RejectLog(
        bad_epochs=np.zeros_like(reject_log.bad_epochs, dtype=bool),
        labels=new_labels,
        ch_names=reject_log.ch_names,
    )

    return new_reject_log


def update_reject_log(orginal_reject_log, bad_epochs_indices, new_label=1):
    """
    Update bad epochs and labels in the RejectLog object.

    Parameters
    ----------
    reject_log : RejectLog
        An instance of the RejectLog class.
    bad_epochs_indices : list of int
        List of epoch indices to be marked as bad.
    new_label : int, optional
        The label to set for the bad epochs in the labels array.
        Default is 1 (bad).

    Returns
    -------
    New instance (copy) of reject_log
    """
    reject_log = copy.deepcopy(orginal_reject_log)
    for idx in bad_epochs_indices:
        if idx < len(reject_log.bad_epochs):
            reject_log.bad_epochs[idx] = True
            reject_log.labels[idx, :] = new_label
        else:
            raise ValueError(
                f"Index {idx} is out of range for bad_epochs of length {len(reject_log.bad_epochs)}"
            )
    return reject_log


def exclude_bad_channels(epochs):
    """Exclude bad channels from the montage (for plotting only good channels)"""
    all_channels = epochs.info["ch_names"]
    bad_channels = epochs.info["bads"]
    good_channels = [ch for ch in all_channels if ch not in bad_channels]
    epochs_good = epochs.copy().pick_channels(good_channels)
    return epochs_good


def plot_specparam_on_scalp(fg, epochs, subject):
    """Plot aperiodc values and its goodness of fit across the scalp"""
    # Extract aperiodic exponent values
    exps = fg.get_params("aperiodic_params", "exponent")
    r_squared = fg.get_params("r_squared")

    # Assuming 'exps' is your data array and 'raw' is an MNE raw object
    # Also, assuming 'unit_label' and 'fontsize' variables are defined

    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # The 'cmap' parameter expects a colormap object, not a string
    im1, _ = mne.viz.plot_topomap(
        exps, epochs.info, axes=axs[0], cmap="viridis", contours=0, show=False
    )
    axs[0].set_title("Exponent Values")

    #  Colorbar setup for the first subplot at the bottom
    cbar_ax1 = fig.add_axes((0.1, 0.05, 0.35, 0.03))
    fig.colorbar(im1, cax=cbar_ax1, orientation="horizontal")

    # Plot the 'errors' data in the second subplot
    im2, _ = mne.viz.plot_topomap(
        r_squared,
        epochs.info,
        axes=axs[1],
        cmap="plasma",
        contours=0,
        show=False,
    )
    axs[1].set_title("R_squared Values")

    # Colorbar setup for the second subplot at the bottom
    cbar_ax2 = fig.add_axes((0.55, 0.05, 0.35, 0.03))
    fig.colorbar(im2, cax=cbar_ax2, orientation="horizontal")
    fig.suptitle(f"sub-{subject} - Exponent and R_squared values")

    plt.show()


def specparam2pandas(fg):
    """
    Converts a SpectralGroupModel object into a pandas DataFrame, with peak parameters and
    corresponding aperiodic fit information.

    Args:
    -----
    fg : specpramGroup
        The SpectralGroupModel object containing the fitting results.

    Returns:
    --------
    peaks_df : pandas.DataFrame
        A DataFrame with the peak parameters and corresponding aperiodic fit information.
        The columns are:
        - 'CF': center frequency of each peak
        - 'PW': power of each peak
        - 'BW': bandwidth of each peak
        - 'error': fitting error of the aperiodic component
        - 'r_squared': R-squared value of the aperiodic fit
        - 'exponent': exponent of the aperiodic component
        - 'offset': offset of the aperiodic component
        - 'knee': knee parameter of the aperiodic component [if is initially present in the fg object]
    Notes:
    ------
    This function creates two DataFrames. The first DataFrame `specparam_aperiodic`
    contains the aperiodic fit information and is based on the `aperiodic_params`
    attribute of the SpectralGroupModel object. The columns are inferred using the
    `get_ap_indices()` and `infer_ap_func()` functions from the specparam package.
    The second DataFrame `peak_df` contains the peak parameters and is based on the
    `peak_params` attribute of the SpectralGroupModel object. The column names are renamed
    to match the headers of `fooof_aperiodic`, and the 'ID' column is cast to integer.
    The two DataFrames are then merged based on a shared 'ID' column.
    """

    specparam_aperiodic = (
        pd.DataFrame(
            fg.get_params("aperiodic_params"),
            columns=get_ap_indices(
                infer_ap_func(np.transpose(fg.get_params("aperiodic_params")))
            ),
        )
        .assign(error=fg.get_params("error"), r_squared=fg.get_params("r_squared"))
        .reset_index(names=["ID"])
    )
    return (
        pd.DataFrame(fg.get_params("peak_params"))  # prepare peaks dataframe
        .set_axis(["CF", "PW", "BW", "ID"], axis=1)  # rename cols
        .astype({"ID": int})
        .join(specparam_aperiodic.set_index("ID"), on="ID")
    )


def examine_spectra(fg, subject):
    """Compare the power spectra between low and high exponent channels"""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    def argmedian(arr):
        return np.argsort(arr)[len(arr) // 2]

    exps = fg.get_params("aperiodic_params", "exponent")
    r_squared = fg.get_params("r_squared")
    spectra_exp = [
        fg.get_model(np.argmin(exps)).power_spectrum,
        fg.get_model(argmedian(exps)).power_spectrum,
        fg.get_model(np.argmax(exps)).power_spectrum,
    ]

    labels_spectra_exp = [
        f"Low Exponent {format(np.min(exps), '.2f')}",
        f"Median Exponent {format(np.median(exps), '.2f')}",
        f"High Exponent {format(np.max(exps), '.2f')}",
    ]

    plot_spectra(
        fg.freqs,
        spectra_exp,
        ax=ax[0],
        labels=labels_spectra_exp,
    )

    spectra_r_squared = [
        fg.get_model(np.argmin(r_squared)).power_spectrum,
        fg.get_model(argmedian(r_squared)).power_spectrum,
        fg.get_model(np.argmax(r_squared)).power_spectrum,
    ]

    labels_spectra_r_squared = [
        f"Low R_squared  {format(np.min(r_squared), '.2f')}",
        f"Median R_squared {format(np.median(r_squared), '.2f')}",
        f"High R_squared {format(np.max(r_squared), '.2f')}",
    ]

    my_colors = ["blue", "green", "red"]
    plot_spectra(
        fg.freqs,
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
        f"sub-{subject} - Power spectra comparison between low, median and high exponent and R_squared values"
    )


def plot_spectra_models_generalized(fg, data, data_type="exps"):
    """
    Plots spectra models based on specified data type (experimental data or r_squared).

    Parameters:
    - fg: The FOOOFGroup object.
    - data: Array-like, data to determine models (experimental data or r_squared).
    - labels: Labels for each plotted spectra model.
    - data_type: Type of data to plot ('exps' for experimental data, 'r_squared' for r_squared values).
    """

    # Define a helper function for median
    def argmedian(data):
        return np.argsort(data)[len(data) // 2]

    # Select the appropriate data for model generation
    if data_type == "exps":
        indices = [np.argmin(data), argmedian(data), np.argmax(data)]
        labels = [
            f"Low {data_type} {format(np.min(data), '.2f')}",
            f"Median {data_type} {format(np.median(data), '.2f')}",
            f"High {data_type} {format(np.max(data), '.2f')}",
        ]
    elif data_type == "r_squared":
        indices = [np.argmin(data), argmedian(data), np.argmax(data)]
        labels = [
            f"Low R_squared {format(np.min(data), '.2f')}",
            f"Median R_squared {format(np.median(data), '.2f')}",
            f"High R_squared {format(np.max(data), '.2f')}",
        ]
    else:
        raise ValueError("data_type must be 'exps' or 'r_squared'")

    # Generate models based on the selected data
    spectra_models = [fg.get_model(idx, regenerate=True) for idx in indices]

    # Iterate over each model and its corresponding label
    for model, label in zip(spectra_models, labels):
        # Print results and plot extracted model fit
        model.print_results()
        model.plot()
        print(label)


def extract_elements(df, subject_id, column_name):
    """
    Extract numerical elements from a specified column for a given subject in a DataFrame.

    Args:
    df (pandas.DataFrame): The DataFrame containing the data.
    subject_id (int): The ID of the subject to extract data for.
    column_name (str): The name of the column to extract data from.

    Returns:
    list: A list of integers extracted from the specified column for the given subject.
        Returns an empty list if no numbers are found or if the cell is empty/NaN.

    Example:
    >>> df = pd.read_excel("data.xlsx")
    >>> elements = extract_elements(df, 104, "ICA_3Take")
    >>> print(elements)
    [1, 2, 3, 4]  # Example output
    """
    # Find the row for the subject
    row = df.loc[df["ID"] == subject_id, ["ID", column_name]]

    # Extract the string from the specified column
    elements_string = row[column_name].values[0]

    # Initialize an empty list for the elements
    elements = []

    # Extract numbers from the string if it's not NaN
    if pd.notna(elements_string):
        elements = [
            int(float(num))
            for num in re.findall(r"(?<!\d)-?\d+(?:\.\d+)?(?!\d)", str(elements_string))
        ]
    return elements


def save_specparam_results(
    fg, epochs_ar, ica, subject, n_interpolated_channels, path=None
):
    """Save the results of the spectral parameterization to a CSV file"""
    channel_names = epochs_ar.info["ch_names"]
    df_channels = pd.DataFrame({"ID": range(len(channel_names)), "ch": channel_names})

    df = specparam2pandas(fg)
    df = df.merge(df_channels, on="ID")
    df["sub_id"] = subject

    # Get the current date and time
    now = datetime.datetime.now()
    df["timestamp"] = now
    df["nr_intepolated_channels"] = n_interpolated_channels
    df["nr_dropped_ica"] = len(ica.exclude)
    df["nr_retained_epochs"] = len(epochs_ar)
    df["nr_retained_ica"] = ica.n_components_ - len(ica.exclude)
    # Create a new list of column names
    cols = ["ch"] + [col for col in df.columns if col != "ch"]

    # Reorder the columns
    df = df[cols]
    if path is None:
        path = f"specparam/sub-{subject}-specparam.csv"
    df.to_csv(path, index=False)
    print(f"Subject {subject} done")
    return df


def plot_models(fg, param_choice="exponent"):
    """
    Plot models from a FOOOF group object based on exponent or R-squared values.

    This function generates three plots (low, median, and high) for the specified
    parameter, prints the results for each model, and displays the corresponding label.

    Parameters:
    -----------
    fg : FOOOFGroup
        The FOOOF group object containing the models to plot.
    param_choice : str, optional
        The parameter to use for selecting models. Must be either 'exponent' or 'r_squared'.
        Default is 'exponent'.

    Raises:
    -------
    ValueError
        If param_choice is not 'exponent' or 'r_squared'.
    """
    if param_choice.lower() == "exponent":
        param = fg.get_params("aperiodic_params", "exponent")
        param_name = "Exponent"
    elif param_choice.lower() == "r_squared":
        param = fg.get_params("r_squared")
        param_name = "R-squared"
    else:
        raise ValueError("param_choice must be either 'exponent' or 'r_squared'")

    def argmedian(arr):
        return np.argsort(arr)[len(arr) // 2]

    labels_spectra = [
        f"Low {param_name} {format(np.min(param), '.2f')}",
        f"Median {param_name} {format(np.median(param), '.2f')}",
        f"High {param_name} {format(np.max(param), '.2f')}",
    ]

    spectra_models = [
        fg.get_model(np.argmin(param), regenerate=True),
        fg.get_model(argmedian(param), regenerate=True),
        fg.get_model(np.argmax(param), regenerate=True),
    ]

    for fm, label in zip(spectra_models, labels_spectra):
        fm.print_results()
        fm.plot()
        plt.title(label)
        plt.show()
        print(label)
