import numpy as np
import mne
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import autoreject
from pyprep.find_noisy_channels import NoisyChannels
from meegkit import dss
import copy



def identify_and_handle_reference_channel(raw: mne.io.Raw, 
                                        reference_names: List[str] = ['VREF', 'Cz', 'REF']) -> mne.io.Raw:
    """
    Identify and properly handle reference channels in EEG data.
    
    This function solves a common problem in EEG preprocessing where the reference
    channel has near-zero values, causing numerical instabilities.
    
    The solution involves:
    1. Identifying potential reference channels by name or by checking for near-zero variance
    2. Temporarily excluding them from processing
    3. Re-referencing to a proper reference (average of all good channels)
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw EEG data
    reference_names : List[str]
        Common names for reference channels to check
        
    Returns
    -------
    raw : mne.io.Raw
        Data with reference channel properly handled
    """
    
    print("\n=== Checking for reference channel issues ===")
    
    # First, let's identify channels with suspiciously low variance
    # These are often reference channels or disconnected channels
    data = raw.get_data(picks='eeg')
    channel_vars = np.var(data, axis=1)
    
    # Find channels with variance less than 1% of median variance
    median_var = np.median(channel_vars[channel_vars > 0])  # Exclude zeros from median
    low_var_threshold = median_var * 0.01
    
    low_var_indices = np.where(channel_vars < low_var_threshold)[0]
    low_var_channels = [raw.ch_names[i] for i in low_var_indices if raw.ch_names[i] not in raw.info['bads']]
    
    if low_var_channels:
        print(f"Found {len(low_var_channels)} channels with very low variance: {low_var_channels}")
        print("These might be reference channels or disconnected channels")
    
    # Check for known reference channel names
    reference_found = None
    for ref_name in reference_names:
        if ref_name in raw.ch_names:
            reference_found = ref_name
            print(f"Found known reference channel: {ref_name}")
            break
    
    # Handle the reference channel
    if reference_found or low_var_channels:
        # Create a copy to avoid modifying the original
        raw = raw.copy()
        
        # Mark reference/low variance channels as bad temporarily
        channels_to_mark = []
        if reference_found:
            channels_to_mark.append(reference_found)
        channels_to_mark.extend(low_var_channels)
        
        # Remove duplicates and update bads
        channels_to_mark = list(set(channels_to_mark))
        raw.info['bads'].extend([ch for ch in channels_to_mark if ch not in raw.info['bads']])
        
        print(f"Temporarily marking as bad: {channels_to_mark}")
        
        # Now re-reference to average, excluding bad channels
        print("Re-referencing to average (excluding bad channels)...")
        raw.set_eeg_reference(ref_channels='average', projection=False)
        
        # After re-referencing, we can remove these from bads if they were only
        # bad due to being reference channels
        if reference_found and reference_found in raw.info['bads']:
            # Check if the channel now has reasonable variance after re-referencing
            ref_idx = raw.ch_names.index(reference_found)
            new_var = np.var(raw.get_data(picks=[ref_idx]))
            
            if new_var > low_var_threshold:
                print(f"Reference channel {reference_found} now has normal variance after re-referencing")
                raw.info['bads'].remove(reference_found)
    
    return raw

def detect_bad_channels_comprehensive(
    raw: mne.io.Raw,
    n_neighbors: int = 5,
    use_pyprep: bool = True,
    use_autoreject_epochs: bool = True,
    epoch_threshold: float = 0.4,
    high_freq_filter: float = 100.0,
    low_freq_filter: float = 1.0
) -> Tuple[List[str], Dict[str, List[str]]]:
    """
    Comprehensive bad channel detection using multiple methods.
    
    This follows the approach recommended in:
    - Pedroni et al. (2019): "Automagic: Standardized preprocessing of big EEG data"
    - Discussion on MNE forum about combining PyPREP with other methods:
      https://mne.discourse.group/t/best-practices-for-bad-channel-detection/
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data (should be minimally processed - only line noise removed)
    n_neighbors : int
        Number of neighbors for LOF algorithm
    use_pyprep : bool
        Whether to use PyPREP RANSAC method
    use_autoreject_epochs : bool
        Whether to use epoch-based rejection
    epoch_threshold : float
        Threshold for marking channels bad based on rejected epochs
    high_freq_filter : float
        High-frequency cutoff for temporary filtering
    low_freq_filter : float
        Low-frequency cutoff for temporary filtering
        
    Returns
    -------
    all_bad_channels : List[str]
        Combined list of all bad channels
    bad_channels_by_method : Dict[str, List[str]]
        Dictionary with bad channels identified by each method
    """
    
    bad_channels_by_method = {}
    
    # Create a temporary filtered copy for bad channel detection
    # This is recommended to avoid low-frequency drifts affecting detection
    print("Creating temporary filtered copy for bad channel detection...")
    raw_temp = raw.copy().filter(
        l_freq=low_freq_filter, 
        h_freq=high_freq_filter,
        picks='eeg',
        n_jobs=-1,
        verbose=False
    )
    
    # Method 1: MNE's Local Outlier Factor (LOF)
    print("\n1. Running Local Outlier Factor detection...")
    try:
        bad_lof = mne.preprocessing.find_bad_channels_lof(
            raw_temp, 
            n_neighbors=n_neighbors,
            threshold=2.5  # More conservative than default
        )[0]
        bad_channels_by_method['LOF'] = bad_lof
        print(f"   LOF found {len(bad_lof)} bad channels: {bad_lof}")
    except Exception as e:
        print(f"   LOF failed: {e}")
        bad_channels_by_method['LOF'] = []
    
    # Method 2: PyPREP (if requested)
    if use_pyprep:
        print("\n2. Running PyPREP RANSAC detection...")
        try:
            # Downsample for PyPREP as recommended
            raw_pyprep = raw_temp.copy().resample(125, npad="auto", verbose=False)
            
            nd = NoisyChannels(raw_pyprep, random_state=42)
            nd.find_bad_by_ransac(
                n_samples=50,
                sample_prop=0.25,
                corr_thresh=0.75,
                frac_bad=0.4,
                corr_window_secs=5.0,
                channel_wise=True,
                max_chunk_size=None
            )
            
            bad_ransac = nd.get_bads(verbose=False)
            bad_channels_by_method['RANSAC'] = bad_ransac
            print(f"   RANSAC found {len(bad_ransac)} bad channels: {bad_ransac}")
        except Exception as e:
            print(f"   RANSAC failed: {e}")
            bad_channels_by_method['RANSAC'] = []
    
    # Method 3: Epoch-based detection using autoreject
    if use_autoreject_epochs:
        print("\n3. Running epoch-based bad channel detection...")
        try:
            # Create short epochs for channel assessment
            epochs = create_analysis_epochs(raw_temp, length=1.0, overlap=0.5)
            
            # Use autoreject to get channel-wise rejection stats
            ar = autoreject.AutoReject(
                n_interpolate=[1],  # Only assess, don't interpolate
                consensus=[0.35],
                thresh_method='bayesian_optimization',
                random_state=42,
                n_jobs=-1,
                verbose=False
            )
            
            ar.fit(epochs)
            reject_log = ar.get_reject_log(epochs)
            
            # Calculate bad channel rate per channel
            bads = np.logical_or(reject_log.labels == 1, reject_log.labels == 2)
            bad_rate_per_channel = bads.mean(axis=0)
            
            # Find channels that are bad in more than threshold of epochs
            bad_epochs_based = [
                ch for ch, rate in zip(epochs.ch_names, bad_rate_per_channel) 
                if rate > epoch_threshold
            ]
            
            bad_channels_by_method['Epochs'] = bad_epochs_based
            print(f"   Epoch-based found {len(bad_epochs_based)} bad channels: {bad_epochs_based}")
            
        except Exception as e:
            print(f"   Epoch-based detection failed: {e}")
            bad_channels_by_method['Epochs'] = []
    
    # Combine all bad channels
    all_bad_channels = list(set(
        ch for channels in bad_channels_by_method.values() 
        for ch in channels
    ))
    
    print(f"\nTotal unique bad channels found: {len(all_bad_channels)}")
    print(f"Bad channels: {sorted(all_bad_channels)}")
    
    return all_bad_channels, bad_channels_by_method


def detect_bad_segments_adaptive(
    raw: mne.io.Raw,
    bad_channels: List[str],
    epoch_length: float = 1.0,
    overlap: float = 0.5,
    consensus_threshold: float = 0.35,
    contamination_threshold: float = 0.1
) -> mne.Annotations:
    """
    Detect bad segments using adaptive epoch-based approach.
    
    Based on:
    - Jas et al. (2017): "Autoreject: Automated artifact rejection for MEG and EEG data"
    - https://mne.discourse.group/t/marking-bad-segments-vs-epochs/
    
    Parameters
    ----------
    raw : mne.io.Raw
        Raw data with bad channels already marked
    bad_channels : List[str]
        List of bad channels to exclude from analysis
    epoch_length : float
        Length of epochs for segment detection
    overlap : float
        Overlap between epochs
    consensus_threshold : float
        Consensus parameter for autoreject
    contamination_threshold : float
        Threshold for marking entire segment as bad
        
    Returns
    -------
    bad_annotations : mne.Annotations
        Annotations marking bad segments
    """
    
    print(f"\nDetecting bad segments using {epoch_length}s epochs...")
    
    # Mark bad channels
    raw_marked = raw.copy()
    raw_marked.info['bads'] = bad_channels
    
    # Create overlapping epochs for finer temporal resolution
    events = mne.make_fixed_length_events(
        raw_marked, 
        duration=epoch_length, 
        overlap=overlap
    )
    
    epochs = mne.Epochs(
        raw_marked,
        events,
        tmin=0,
        tmax=epoch_length,
        baseline=None,
        preload=True,
        reject_by_annotation=False,  # We're creating annotations
        picks='eeg'
    )
    
    # Use autoreject with conservative settings
    ar = autoreject.AutoReject(
        n_interpolate=[0],  # Don't interpolate, just mark
        consensus=[consensus_threshold],
        thresh_method='bayesian_optimization',
        random_state=42,
        n_jobs=-1,
        verbose=False
    )
    
    # Fit autoreject
    ar.fit(epochs)
    reject_log = ar.get_reject_log(epochs)
    
    # Find segments where many channels are bad
    bad_segments = []
    for idx, is_bad_epoch in enumerate(reject_log.bad_epochs):
        if is_bad_epoch:
            # Check the contamination level
            epoch_labels = reject_log.labels[idx]
            contamination = np.mean(epoch_labels > 0)
            
            if contamination > contamination_threshold:
                onset = events[idx, 0] / raw.info['sfreq']
                bad_segments.append({
                    'onset': onset,
                    'duration': epoch_length,
                    'description': f'bad_segment_cont{contamination:.2f}'
                })
    
    # Merge overlapping segments
    merged_segments = merge_overlapping_segments(bad_segments)
    
    # Create annotations
    if merged_segments:
        onsets = [seg['onset'] for seg in merged_segments]
        durations = [seg['duration'] for seg in merged_segments]
        descriptions = ['bad_segment'] * len(merged_segments)
        
        bad_annotations = mne.Annotations(
            onset=onsets,
            duration=durations,
            description=descriptions,
            orig_time=raw.info.get('meas_date')
        )
    else:
        bad_annotations = mne.Annotations([], [], [])
    
    print(f"Found {len(bad_annotations)} bad segments")
    print(f"Total bad time: {sum(bad_annotations.duration):.1f}s "
          f"({sum(bad_annotations.duration)/raw.times[-1]*100:.1f}% of data)")
    
    return bad_annotations
# Helper functions

def create_analysis_epochs(raw: mne.io.Raw, length: float = 1.0, overlap: float = 0.5):
    """Create epochs for analysis with proper parameters."""
    events = mne.make_fixed_length_events(
        raw, 
        duration=length, 
        overlap=overlap
    )
    
    epochs = mne.Epochs(
        raw,
        events,
        tmin=0,
        tmax=length,
        baseline=None,
        detrend=1,  # Linear detrend
        preload=True,
        reject_by_annotation=True
    )
    
    return epochs


def merge_overlapping_segments(segments: List[Dict]) -> List[Dict]:
    """Merge overlapping bad segments."""
    if not segments:
        return []
    
    # Sort by onset
    segments = sorted(segments, key=lambda x: x['onset'])
    
    merged = [segments[0].copy()]
    
    for current in segments[1:]:
        last = merged[-1]
        
        # Check if segments overlap or are adjacent
        if current['onset'] <= last['onset'] + last['duration']:
            # Merge segments
            end_last = last['onset'] + last['duration']
            end_current = current['onset'] + current['duration']
            last['duration'] = max(end_last, end_current) - last['onset']
        else:
            merged.append(current.copy())
    
    return merged
