import numpy as np
import autoreject

from .epochs import create_epochs

def detect_bad_epochs_strict(raw, length, overlap, threshold):
    """
    Implements the 40% bad channel threshold rule.
    Returns:
        bad_mask (np.array of bool): True if epoch is BAD
        epochs (mne.Epochs): The epochs object used for detection
    """
    print(f"   -> Segmenting: {length}s epochs, {overlap}s overlap...")
    # Note: MNE 'overlap' parameter is the duration of overlap
    epochs = create_epochs(raw, length=length, overlap=overlap)

    print("   -> Running Autoreject (to find bad channels per epoch)...")
    # We use a permissive consensus to just get the error logs
    ar = autoreject.AutoReject(consensus=[0.5], verbose=False)
    ar.fit(epochs)
    reject_log = ar.get_reject_log(epochs)

    # Calculate ratio of bad channels per epoch
    # reject_log.labels shape: (n_epochs, n_channels)
    # 0=Good, 1=Bad, 2=Bad(Interpolated) -> We count 1 & 2 as bad
    n_consistently_bad = np.sum(reject_log.labels != 0, axis=1)
    n_channels = len(epochs.ch_names)
    bad_ratios = n_consistently_bad / n_channels

    # Apply strict threshold
    bad_mask = bad_ratios > threshold

    print(f"   -> Detection: {sum(bad_mask)}/{len(epochs)} epochs rejected "
          f"({sum(bad_mask) / len(epochs):.1%}) (> {threshold * 100}% bad chans)")

    return bad_mask, epochs