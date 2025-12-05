import numpy as np
import mne
from pyriemann.clustering import Potato
from pyriemann.estimation import Covariances
from pyprep.find_noisy_channels import NoisyChannels
from typing import List
import logging
from typing import List, Dict, Union, Tuple, Optional

# Configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def run_pyprep_cleaning(
    raw: mne.io.Raw,
    output_mode: str = "all",
    resample_freq: float = 125.0,
    random_state: int = 1337,
    picks: str = "eeg",
) -> Dict[str, List[str]]:
    """
    Executes the PyPREP pipeline to identify bad EEG channels.

    Best Practice:
    - This function operates on a downsampled copy of the data to speed up
      calculations (specifically RANSAC), which is computationally expensive.
    - It verifies that a montage is present, as RANSAC relies on sensor geometry.

    Parameters
    ----------
    raw : mne.io.Raw
        The continuous data to analyze.
    output_mode : str
        'all' runs the full suite (Correction, Deviation, RANSAC, HF Noise).
        Otherwise runs the manual configuration defined in the else block.
    resample_freq : float
        Frequency to downsample to for processing speed (default 125Hz).
    random_state : int
        Seed for reproducibility.
    picks : str
        Data types to include. Defaults to 'eeg' to exclude EOG/Stim channels.

    Returns
    -------
    Dict[str, List[str]]
        A dictionary mapping the detection method to a list of bad channel names.
    """
    # 1. Validation: Ensure montage exists for RANSAC
    if raw.get_montage() is None:
        raise ValueError(
            "No montage found. PyPREP (specifically RANSAC) requires "
            "channel locations. Use raw.set_montage() before running this."
        )

    # 2. Preparation: Pick EEG channels and resample for speed
    # We copy to avoid modifying the original data during resampling
    raw_process = raw.copy().pick(picks)
    if raw_process.info["sfreq"] > resample_freq:
        raw_process.resample(resample_freq)

    logger.info(
        f"Running PyPREP on {len(raw_process.ch_names)} channels at {resample_freq}Hz."
    )

    # 3. Initialization
    # Suppress MNE info messages during the loop to reduce clutter
    noisy_detector = NoisyChannels(raw_process, random_state=random_state)

    # 4. Execution
    if output_mode == "all":
        noisy_detector.find_all_bads(
            ransac=True, channel_wise=True, max_chunk_size=None
        )
    else:
        # Manual configuration from original script
        noisy_detector.find_bad_by_correlation(
            correlation_secs=1.0, correlation_threshold=0.4, frac_bad=0.01
        )
        noisy_detector.find_bad_by_deviation(deviation_threshold=5.0)
        noisy_detector.find_bad_by_ransac(
            n_samples=50,
            sample_prop=0.25,
            corr_thresh=0.75,
            frac_bad=0.4,
            corr_window_secs=5.0,
            channel_wise=True,
            max_chunk_size=None,
        )

    # 5. Retrieval
    bads_summary = noisy_detector.get_bads(verbose=True, as_dict=True)

    # Calculate unique total bads for logging
    all_bads_unique = set([ch for ch_list in bads_summary.values() for ch in ch_list])
    logger.info(f"PyPREP identified {len(all_bads_unique)} unique bad channels.")

    return bads_summary


def annotate_bads_to_raw(
    raw: mne.io.Raw, bads_dict: Dict[str, List[str]], inplace: bool = False
) -> Tuple[mne.io.Raw, Dict[str, str]]:
    """
    Updates the raw object's info['bads'] based on PyPREP output.

    Parameters
    ----------
    raw : mne.io.Raw
        The original data.
    bads_dict : Dict
        Output from run_pyprep_cleaning.
    inplace : bool
        If True, modifies the raw object directly (saves memory).
        If False, creates a copy (safer for testing).

    Returns
    -------
    Tuple[mne.io.Raw, Dict[str, str]]
        The updated raw object and a dictionary mapping channel names
        to the reason they were rejected.
    """
    if inplace:
        raw_annotated = raw
    else:
        raw_annotated = raw.copy()

    # Consolidate bads and track reasons
    new_bads_set = set()
    bad_reasons = {}

    for error_type, channels in bads_dict.items():
        if channels:
            # Format: 'bad_by_hf_noise' -> 'High Frequency Noise'
            readable_reason = (
                error_type.replace("bad_by_", "").replace("_", " ").title()
            )

            for ch in channels:
                new_bads_set.add(ch)
                # Store reason. Note: Overwrites if channel fails multiple checks.
                # Usually fine as any failure warrants exclusion.
                bad_reasons[ch] = readable_reason

    # Functional Exclusion: Union with existing bads
    current_bads = set(raw_annotated.info["bads"])
    updated_bads = current_bads.union(new_bads_set)

    # Update MNE object
    raw_annotated.info["bads"] = list(updated_bads)

    logger.info(
        f"Raw object updated. Total bad channels: {len(raw_annotated.info['bads'])}"
    )

    return raw_annotated, bad_reasons


def apply_potato_to_raw(raw, force_run=False, **kwargs):
    """
    Runs Potato artifact detection and adds the annotations directly to the raw object.
    Checks if BAD_potato annotations already exist to avoid re-running.

    Parameters
    ----------
    raw : mne.io.Raw
        The raw data to annotate.
    force_run : bool
        If True, runs detection even if BAD_potato annotations exist.
    **kwargs :
        Arguments passed to annotate_with_potato_robust
        (e.g., h_freq=40.0, start_threshold=2.5).

    Returns
    -------
    raw : mne.io.Raw
        The raw object (original or copy depending on logic), now containing the BAD_potato annotations (if processed).
    """
    # 1. Check for existing Potato annotations
    if raw.annotations and not force_run:
        # Check if any annotation description contains 'BAD_potato'
        # We use 'in' because sometimes descriptions might vary slightly
        has_potato = any("BAD_potato" in desc for desc in raw.annotations.description)

        if has_potato:
            print(
                "ℹ️ 'BAD_potato' annotations already exist. Skipping Potato processing."
            )
            return raw

    # 2. Run the detection
    # Note: annotate_with_potato_robust must be imported/defined in your script
    potato_annots = annotate_with_potato_robust(raw, **kwargs)

    # 3. Update the raw object if artifacts were found
    if potato_annots:
        # Create a copy to ensure we don't modify the original in-place unexpectedly
        # if the user intended to keep the original clean.
        # However, for this specific wrapper function intended to "apply to raw",
        # returning a modified copy is a standard safe practice.
        raw_annotated = raw.copy()

        # Combine existing annotations with new Potato ones
        raw_annotated.set_annotations(raw_annotated.annotations + potato_annots)
        print(
            f"✅ Added {len(potato_annots)} 'BAD_potato' annotations to the raw object."
        )
        return raw_annotated

    return raw


def annotate_with_potato_robust(
    raw,
    l_freq=0.5,
    h_freq=None,
    start_threshold=2.9,
    window_duration=1.5,
    resample_freq=250,
):
    # 1. CLEAN UP BEFORE STARTING
    raw_clean = raw.copy()
    raw_clean.resample(resample_freq, verbose=False)
    # C. HIGH-PASS FILTER (The requested addition)
    if l_freq is not None:
        print(f"   High-pass filtering > {l_freq} Hz...")
        raw_clean.filter(l_freq=l_freq, h_freq=h_freq, verbose=False)
    if len(raw_clean.info["bads"]) > 0:
        print(
            f"Interpolating {len(raw_clean.info['bads'])} bad channels before Potato..."
        )
        raw_clean.interpolate_bads(reset_bads=True, verbose=False)

    # 2. EPOCH THE DATA (FORCE KEEP)
    events = mne.make_fixed_length_events(raw_clean, duration=window_duration)

    # reject_by_annotation=False forces MNE to ignore existing BAD labels
    epochs = mne.Epochs(
        raw_clean,
        events,
        tmin=0,
        tmax=window_duration,
        baseline=None,
        preload=True,
        verbose=False,
        reject=None,
        flat=None,
        reject_by_annotation=False,
    )

    if len(epochs) == 0:
        print("Error: Epochs are still empty. Check if your data duration > 0.")
        return None

    picks = mne.pick_types(raw_clean.info, eeg=True, ecg=False, eog=False)
    X = epochs.get_data(picks=picks)

    # 3. COMPUTE COVARIANCES
    try:
        covs = Covariances(estimator="lwf").fit_transform(X)
    except ValueError as e:
        print(f"Error computing covariances: {e}. Data might be too corrupt.")
        return None

    # 4. FIT POTATO (With Auto-Scaling Threshold)
    potato = None
    final_threshold = start_threshold

    for try_thresh in [start_threshold, 5, 10, 20, 50, 100]:
        try:
            print(f"Trying Potato with threshold={try_thresh}...")
            temp_potato = Potato(metric="riemann", threshold=try_thresh)
            temp_potato.fit(covs)

            potato = temp_potato
            final_threshold = try_thresh
            print(f"Potato fit successfully with threshold={final_threshold}")
            break
        except ValueError:
            continue

    if potato is None:
        print("Potato failed even at threshold=100. Your data is extremely noisy.")
        return None

    # 5. PREDICT
    labels = potato.predict(covs)  # 1=Clean, 0=Bad
    bad_indices = np.where(labels == 0)[0]

    onsets = []
    durations = []

    if len(bad_indices) > 0:
        epoch_times = events[:, 0] / raw_clean.info["sfreq"]
        for idx in bad_indices:
            onsets.append(epoch_times[idx])
            durations.append(window_duration)

        print(f"Found {len(bad_indices)} bad segments.")

        return mne.Annotations(
            onset=onsets,
            duration=durations,
            description=["BAD_potato"] * len(onsets),
            orig_time=raw.annotations.orig_time,
        )
    else:
        print("No artifacts found.")
        return None
