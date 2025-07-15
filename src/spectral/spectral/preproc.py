""" Function for preprocessing EEG data """
import mne
import matplotlib.pyplot as plt


from eeglabio.utils import export_mne_raw

from meegkit.detrend import detrend
from meegkit import dss
from pyprep.find_noisy_channels import NoisyChannels

from autoreject import Ransac  # noqa
from typing import  Optional, Union, List
from pathlib import Path
import numpy as np
from mne import pick_types

from .epochs import create_epochs
from .utils import load_config, ProjectPaths



# This part could be replaced by read MFF files (set files are just downsampled MFF files saved to the local laptop drive)
def load_data(subject_id: Union[str, int], 
              data_path: Union[str, Path],
              session: str = "01",
              task: str = "rest") -> mne.io.Raw:
    """
    Load EEG data with explicit path and preprocessing parameters.
    
    This version makes all key parameters visible at the function call,
    which can be helpful for debugging and when you need to override
    default settings for specific subjects.
    
    Args:
        subject_id: Subject identifier (will be zero-padded to 3 digits)
        data_path: Path to the folder containing the BIDS-formatted data of the subject
        session: Session identifier (default "01")
        task: Task name (default "rest")
        
    Returns:
        mne.io.Raw: Loaded and preprocessed raw data
        
    Example:
        >>> raw = load_data(
        ...     subject_id=001,
        ...     data_path="/data/eeg_study/raw_bids/sub-001",
        ... )
    """
    # Convert inputs to proper types
    data_path = Path(data_path)
    
    # Format subject ID consistently
    if isinstance(subject_id, int):
        subject_str = f"{subject_id:03d}"
    else:
        subject_str = str(subject_id).zfill(3)
    
    # Build the file path following BIDS structure
    filename = f"sub-{subject_str}_ses-{session}_task-{task}_eeg.set"
    data_file = data_path  / f"ses-{session}" / "eeg" / filename
    
    # Check if file exists with helpful error message
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n"
            f"Project path: {data_path}\n"
            f"Looking for: {filename}\n"
            f"Please check that the subject data has been copied to the BIDS directory."
        )
    print(f"Loading data from: {data_file}")
    
    # Load the raw data
    raw = mne.io.read_raw_eeglab(data_file, preload=True)
    print(f"Loaded {len(raw.ch_names)} channels, {raw.times[-1]:.1f} seconds of data")
    
    # Handle ECG channels
    ecg_channels = [ch for ch in raw.ch_names if "ECG" in ch.upper()]
    if ecg_channels:
        ecg_mapping = {ch: 'ecg' for ch in ecg_channels}
        raw.set_channel_types(ecg_mapping)
        print(f"Identified ECG channels: {ecg_channels}")
    
    # Set the channel type for 'VREF' to 'misc' before applying montage
        # Remove 'VREF' channel if it exists
    if 'VREF' in raw.ch_names:
        raw.drop_channels(['VREF'])
        print("Removed 'VREF' channel.")
    # Apply montage
    print("Applying GSN-HydroCel-256 montage...")
    montage = mne.channels.make_standard_montage("GSN-HydroCel-256")
    raw.set_montage(montage, on_missing='warn')
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

def reject_log_to_annotations(reject_log, epochs):
    """
    Convert a reject_log from autoreject into MNE annotations.

    This function identifies the time spans of bad epochs and creates
    MNE annotations for them, which can be added back to the raw object.

    Args:
        reject_log: The RejectLog instance from autoreject.
        epochs: The MNE epochs object from which the reject_log was generated.

    Returns:
        mne.Annotations: Annotations object marking the bad segments.
    """
    bad_epoch_indices = np.where(reject_log.bad_epochs)[0]

    if len(bad_epoch_indices) == 0:
        return mne.Annotations(onset=[], duration=[], description=[])

    # Get the event timings from the epochs object
    onsets = epochs.events[bad_epoch_indices, 0] / epochs.info["sfreq"]
    duration = len(epochs.times) / epochs.info["sfreq"]

    # Create annotations for each bad epoch
    bad_annotations = mne.Annotations(
        onset=onsets,
        duration=[duration] * len(onsets),
        description=["bad_autoreject"] * len(onsets),
        orig_time=epochs.info.get('meas_date')
    )

    return bad_annotations