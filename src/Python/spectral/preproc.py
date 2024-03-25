""" Function for preprocessing EEG data """
import mne
import matplotlib.pyplot as plt


from eeglabio.utils import export_mne_raw

from meegkit.detrend import detrend
from meegkit import dss
from pyprep.find_noisy_channels import NoisyChannels

from autoreject import Ransac  # noqa
from typing import List


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
    montage = mne.channels.make_standard_montage("GSN-HydroCel-257")
    raw.set_montage(montage, match_alias={"VREF": "Cz"})
    raw.info["bads"] = ["VREF"]
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


def zapline_clean(raw, fline, method="line", iter_param=None):
    """Apply the zapline cleaning"""
    raw
    data = raw.get_data().T  # Convert mne data to numpy darray
    sfreq = raw.info["sfreq"]  # Extract the sampling freq
    # Apply MEEGkit toolbox function

    if method == "iter":
        temp_data, _ = dss.dss_line_iter(data, fline, sfreq, **iter_param)
        temp_data, _ = dss.dss_line_iter(temp_data, fline, sfreq, **iter_param)
        temp_data, _ = dss.dss_line_iter(temp_data, fline, sfreq, **iter_param)
        out, _ = dss.dss_line_iter(temp_data, fline, sfreq, **iter_param)
    elif method == "line":
        out, _ = dss.dss_line(data, fline, sfreq, nremove=15)

    cleaned_raw = mne.io.RawArray(
        out.T, raw.info
    )  # Convert output to mne RawArray again
    cleaned_raw.set_annotations(raw.annotations)
    return cleaned_raw


def apply_pyprep(raw: mne.io.Raw) -> List[str]:
    """Apply the pyprep cleaning"""
    temp = raw.copy().resample(125)
    nd = NoisyChannels(temp, random_state=1337)
    # nd.find_bad_by_correlation(
    #    correlation_secs=1.0, correlation_threshold=0.4, frac_bad=0.01
    # )
    # nd.find_bad_by_deviation(deviation_threshold=5.0)
    nd.find_all_bads(ransac=True, channel_wise=True, max_chunk_size=None)

    print("bad all", nd.get_bads(verbose=True))
    return [x for x in nd.get_bads(verbose=False)]


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