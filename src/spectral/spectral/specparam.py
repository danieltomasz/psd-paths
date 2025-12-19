import pandas as pd
import numpy as np
from specparam.plts.spectra import plot_spectra
import matplotlib.pyplot as plt


def specparam2pandas(fg):
    """
    Converts a SpectralGroupModel object into a pandas DataFrame, with peak parameters and
    corresponding aperiodic fit information.

    Args:
    -----
    fg : SpectralGroupModel
        The SpectralGroupModel object containing the fitting results.

    Returns:
    --------
    peaks_df : pandas.DataFrame
        A DataFrame with the peak parameters and corresponding aperiodic fit information.
        Each row represents a single peak, with columns:
        - 'CF': center frequency of each peak
        - 'PW': power of each peak
        - 'BW': bandwidth of each peak
        - 'ID': identifier for the spectrum this peak belongs to
        - Aperiodic parameters: 'offset', 'exponent' (and 'knee' if knee mode)
        - 'error_mae': mean absolute error of the fit
        - 'gof_rsquared': R-squared value of the fit

    Notes:
    ------
    This function creates a long-format DataFrame where each peak is a separate row.
    Peaks are joined with their corresponding aperiodic parameters via the 'ID' column.
    If a spectrum has no peaks, it will still appear with NaN values for peak columns.
    """

    # Check if model has been fit
    if not fg.results.has_model:
        raise ValueError("No model fit results available. Please fit the model first.")

    # Extract aperiodic parameters - one row per spectrum
    ap_params = fg.get_params("aperiodic")
    ap_labels = list(fg.modes.aperiodic.params.labels)

    specparam_aperiodic = pd.DataFrame(ap_params, columns=ap_labels)

    # Add metrics
    specparam_aperiodic["error_mae"] = fg.get_metrics("error", "mae")
    specparam_aperiodic["gof_rsquared"] = fg.get_metrics("gof", "rsquared")

    # Add ID column
    specparam_aperiodic = specparam_aperiodic.reset_index(names=["ID"])

    # Extract peak parameters
    peaks = fg.get_params("peak")

    if peaks.size > 0:
        # peaks array has shape (n_peaks, 4) where columns are [CF, PW, BW, ID]
        # The last column is the model index
        peak_df = pd.DataFrame(peaks)
        peak_df.columns = ["CF", "PW", "BW", "ID"]
        peak_df["ID"] = peak_df["ID"].astype(int)

        # Left join peaks with aperiodic parameters
        result = specparam_aperiodic.merge(peak_df, on="ID", how="left")
    else:
        # No peaks found - create empty peak dataframe with proper columns
        peak_df = pd.DataFrame(columns=["CF", "PW", "BW", "ID"])

        # Left join to maintain all spectra with NaN for peak values
        result = specparam_aperiodic.merge(peak_df, on="ID", how="left")

    return result


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
