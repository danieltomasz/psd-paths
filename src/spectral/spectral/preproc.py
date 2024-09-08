""" Function for preprocessing EEG data """
import mne
import matplotlib.pyplot as plt


from eeglabio.utils import export_mne_raw

from meegkit.detrend import detrend
from meegkit import dss
from pyprep.find_noisy_channels import NoisyChannels

from autoreject import Ransac  # noqa
from typing import List
import numpy as np
from mne import pick_types

from .epochs import create_epochs

# This part could be replaced by read MFF files (set files are just downsampled MFF files saved to the local laptop drive)
def load_data(subject: str, project_path: str) -> mne.io.Raw:
    """ Load the data from the dataset """
    fname_path = f"{project_path}/data/raw/BIDS/sub-{
        subject}/ses-01/eeg/sub-{subject}_ses-01_task-rest_eeg.set"
    print(f"Loading data from {fname_path}")
    raw = mne.io.read_raw_eeglab(fname_path, preload=True)
    ch_name_type = dict(zip(raw.ch_names, raw.get_channel_types()))
    ch_name_ecg = [name for name in raw.ch_names if name in "ECG"]
    dict_ecg = {sub[0]: "ecg" for sub in (ele.split() for ele in ch_name_ecg)}
    raw.set_channel_types(dict_ecg)
    montage = mne.channels.make_standard_montage("GSN-HydroCel-256")
    raw.pick_channels([ch for ch in raw.ch_names if ch != "VREF"])
    #raw.set_montage(montage, match_alias={"VREF": "Cz"})
    raw.set_montage(montage)
    #raw.info["bads"] = ["VREF"]
    return raw


def detrending(raw, order=1):
    """Detrend the data"""
    data = raw.get_data().T  # Convert mne data to numpy darray
    data, _, _ = detrend(data, order=order)
    detrended_raw = mne.io.RawArray(
        data.T, raw.info
    )
    detrended_raw.set_annotations(raw.annotations)
    return detrended_raw


def apply_projection(raw):
    """Apply the projection"""
    raw_clean = raw.copy()
    raw_clean.interpolate_bads()
    raw_clean.set_eeg_reference(
        "average", projection=True)  # compute the reference
    raw_ref = raw.copy().add_proj(raw_clean.info["projs"][0])
    return raw_ref.apply_proj()  # apply the reference


def get_bad_lof(raw):
    bad_channels = mne.preprocessing.find_bad_channels_lof(raw)
    # print(bad_channels)
    raw_marked_bad = raw.copy()
    raw_marked_bad.info["bads"].extend(bad_channels)  # add a list of channels
    bad_channels2 = mne.preprocessing.find_bad_channels_lof(raw_marked_bad)
    return bad_channels + bad_channels2

def zapline_clean(raw, fline, ntimes=1, method="line", iter_param=None):
    """Apply the zapline cleaning"""
    raw
    data = raw.get_data().T  # Convert mne data to numpy darray
    sfreq = raw.info["sfreq"]  # Extract the sampling freq
    # Apply MEEGkit toolbox function
    for _ in range(ntimes):
        if method == "iter":
            data, _ = dss.dss_line_iter(data, fline, sfreq, **iter_param)
        elif method == "line":
            data, _ = dss.dss_line(data, fline, sfreq, nremove=15)

    cleaned_raw = mne.io.RawArray(
        data.T, raw.info
    )  # Convert output to mne RawArray again
    cleaned_raw.set_annotations(raw.annotations)
    return cleaned_raw


def apply_pyprep(raw: mne.io.Raw, output: str = "all", as_dict=True) -> List[str]:
    """Apply the pyprep cleaning"""
    temp = raw.copy().resample(125)
    nd = NoisyChannels(temp, random_state=1337)
    #nd.find_bad_by_correlation(
    #    correlation_secs=1.0, correlation_threshold=0.4, frac_bad=0.01
    # )
    #nd.find_bad_by_deviation(deviation_threshold=5.0)
    if output == "all":
        nd.find_all_bads(ransac=True, channel_wise=True, max_chunk_size=None)
        print("bad all", nd.get_bads(verbose=True))
        return nd.get_bads(verbose=True, as_dict=as_dict)
    else:
        nd.find_bad_by_correlation(
         correlation_secs=1.0, correlation_threshold=0.4, frac_bad=0.01
        )
        nd.find_bad_by_deviation(deviation_threshold=5.0)
        nd.find_bad_by_ransac(n_samples=50, sample_prop=0.25, corr_thresh=0.75,
                              frac_bad=0.4, corr_window_secs=5.0,
                              channel_wise=True, max_chunk_size=None)
        return nd.get_bads(verbose=False, as_dict=as_dict)



def get_bad_channels(raw, save_figs=False):
    """Reject bad channels by RANSAC on the epochs"""
    clean_raw_downsampled = raw.copy().resample(125, npad="auto")
    epochs = create_epochs(clean_raw_downsampled)
    # plot_epochs(epochs, stage="epochs", n_epochs=10, n_channels=25)
    ransac = Ransac(verbose=False, n_jobs=-1)
    _ = ransac.fit_transform(epochs)
    print("\n".join(ransac.bad_chs_))
    #sensor_plot = raw.plot_sensors(show_names=True)
    #if save_figs:
    #    sensor_plot.savefig(
    #        f"{figure_path}/sub-{subject}_bad_sensors.png", dpi=300, bbox_inches="tight"
    #    )
    return [x for x in ransac.bad_chs_]

def compute_ptp_matrix(data, epoch_duration, sfreq):
    """
    Compute peak-to-peak amplitudes for 1-second epochs across all channels.

    Args:
        data (np.ndarray): EEG data with shape (n_channels, n_times)
        epoch_duration (float): Duration of each epoch in seconds
        sfreq (float): Sampling frequency of the data

    Returns:
        np.ndarray: Matrix of peak-to-peak amplitudes with shape (n_epochs, n_channels)
    """
    n_channels, n_times = data.shape
    n_samples_per_epoch = int(epoch_duration * sfreq)
    n_epochs = n_times // n_samples_per_epoch

    # Reshape data into 3D array (n_epochs, n_channels, n_samples_per_epoch)
    epoched_data = np.reshape(
        data[:, : n_epochs * n_samples_per_epoch],
        (n_epochs, n_channels, n_samples_per_epoch),
    )

    # Compute peak-to-peak amplitudes across time for each epoch and channel
    ptp_matrix = np.ptp(epoched_data, axis=2)

    return ptp_matrix


def get_bad_annotations(
    raw_filtered, peak_threshold=2.5, epoch_duration=1.0, extend_bad_duration=1.5
):
    """get bad annotations with high peak-to-peak amplitude"""
    picks = pick_types(raw_filtered.info, meg=False, eeg=True, exclude="bads")

    # Assume 'raw' is your mne.io.Raw object
    data = raw_filtered.get_data(picks=picks)
    sfreq = raw_filtered.info["sfreq"]

    # Compute peak-to-peak amplitude matrix
    ptp_matrix = compute_ptp_matrix(data, epoch_duration, sfreq)
    # print(ptp_matrix)
    peak = peak_threshold * np.median(ptp_matrix)

    print(peak)

    annotations, bads = mne.preprocessing.annotate_amplitude(
        raw_filtered,
        peak=peak,
        flat=None,
        bad_percent=5,
        min_duration=0.005,
        picks=picks,
        verbose=True,
    )

    sfreq = raw_filtered.info["sfreq"]  # Sampling frequency
    # Number of samples to extend
    extend_samples = int(extend_bad_duration * sfreq)

    updated_annotations = []
    for ann in annotations:
        onset = ann["onset"]
        duration = ann["duration"]
        description = ann["description"]

        new_onset = max(0, onset - extend_samples / sfreq)
        new_duration = duration + 2 * extend_samples / sfreq
        updated_annotations.append(
            {
                "onset": new_onset,
                "duration": new_duration,
                "description": description,
                "orig_time": None,
            }
        )

    onset = [ann["onset"] for ann in updated_annotations]
    duration = [ann["duration"] for ann in updated_annotations]
    description = [ann["description"] for ann in updated_annotations]
    return mne.Annotations(onset, duration, description, orig_time=None)