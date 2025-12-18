import mne
import autoreject
from mne import Epochs
import numpy as np
import matplotlib.pyplot as plt

def create_epochs(
    raw: mne.io.Raw,
    length: float = 5,
    overlap: float = 1.5,
    reject_by_annotation = True, 
    epochs_params: dict | None = None,
):
    """Create epochs from raw data"""
    # Create fixed length events
    events = mne.make_fixed_length_events(
        raw=raw, 
        start=0, 
        stop=None, 
        duration=length, 
        overlap=overlap, 
        first_samp=False  # More intuitive timing
    )

    # Default parameters for epochs
    default_params = {
        "raw": raw,
        "events": events,
        "tmin": 0,
        "tmax": length,
        "detrend": 1,
        "baseline": None,
        "preload": True,
        "reject_by_annotation": reject_by_annotation,  # CRITICAL: Use PyPrep annotations
        "reject": None,  # You'll use Autoreject instead
    }
    
    # Update with custom parameters if provided
    if epochs_params is not None:
        default_params.update(epochs_params)

    return mne.Epochs(**default_params)

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


def get_reject_log(epochs, resample=None, consensus=[0.8], n_interpolate=None):
    """Get reject log from epochs"""
    if n_interpolate is None:
        n_interpolate = [1, 2, 16, 32, 64, 128]
    if resample:
        eeg_epochs = epochs.copy().resample(resample)
    else:
        eeg_epochs = epochs.copy()
    auto_reject_pre_ica = autoreject.AutoReject(
        n_interpolate=n_interpolate,
        n_jobs=-1,
        random_state=100,
        thresh_method="bayesian_optimization",
        verbose=False,
        # n_interpolate=np.array([0]),
        consensus=consensus,
    ).fit(eeg_epochs)
    print("fitting finished")
    _, reject_log = auto_reject_pre_ica.transform(eeg_epochs, return_log=True)
    # print(reject_log.bad_epochs)
    return reject_log



def visualise_bad_epochs(reject_log):
    """Visualise the bad epochs and channels."""
    bads = np.logical_or(reject_log.labels == 1, reject_log.labels == 2)
    plt.imshow(bads, cmap="viridis")
    plt.colorbar(orientation="horizontal", pad=0.1)
    plt.show()

    print(f"Currently removed number of epochs {np.sum(reject_log.bad_epochs)}")
    # print(bads)
    # print(bads.shape)
    good_epochs_percentage = (1 - bads.mean(axis=1)) * 100

    # print("Percentage of bad epochs in each epoch:")
    # display(good_epochs_percentage)

    print("Percentage of good epochs in each  candidate for removal epoch:")
    for i in range(0, len(good_epochs_percentage)):
        if good_epochs_percentage[i] < 75:
            print(f"Epoch {i}: {good_epochs_percentage[i]:.2f}%")
            # print(f"Epoch {i}: {good_epochs_percentage[i]:.2f}%")


def plot_rejection_matrix(
    rejection_log,
    subject: int,
    threshold: float = 0.75,
    title: str | None = None
) -> plt.Figure:
    """
    Visualizes the rejection log as a matrix with quality metrics.
    Shows which channels were bad in which epochs and highlights problematic epochs.
    
    Parameters
    ----------
    rejection_log : RejectLog
        The autoreject RejectLog object
    subject : int
        Subject identifier
    threshold : float
        Percentage threshold for marking epochs as problematic (default: 0.75 = 75%)
    title : str, optional
        Custom title for the plot
    
    Returns
    -------
    plt.Figure
        The matplotlib figure
    """
    if title is None:
        title = f'Rejection Log - Subject {subject}'
    
    # Combine bad (1) and interpolated (2) into single "bad" category
    bads = np.logical_or(rejection_log.labels == 1, rejection_log.labels == 2)
    
    # Calculate percentage of good channels per epoch
    good_epochs_percentage = (1 - bads.mean(axis=1)) * 100
    
    # Identify problematic epochs
    problematic_epochs = good_epochs_percentage < (threshold * 100)
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), 
                                     gridspec_kw={'height_ratios': [3, 1]})
    
    # --- Top plot: Rejection matrix ---
    im = ax1.imshow(
        bads.T,  # Transpose: channels on Y, epochs on X
        aspect='auto',
        cmap='viridis',
        interpolation='nearest'
    )
    
    ax1.set_xlabel('Epoch Number')
    ax1.set_ylabel('Channel')
    ax1.set_title(title)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax1, orientation='horizontal', pad=0.1)
    cbar.set_label('Bad (1) / Good (0)')
    
    # Set y-ticks to show channel names
    n_channels = len(rejection_log.ch_names)
    if n_channels <= 20:
        ax1.set_yticks(range(n_channels))
        ax1.set_yticklabels(rejection_log.ch_names, fontsize=8)
    else:
        step = max(1, n_channels // 20)
        ax1.set_yticks(range(0, n_channels, step))
        ax1.set_yticklabels(rejection_log.ch_names[::step], fontsize=8)
    
    # Highlight problematic epochs with vertical lines
    for epoch_idx in np.where(problematic_epochs)[0]:
        ax1.axvline(epoch_idx, color='red', alpha=0.3, linewidth=2)
    
    # --- Bottom plot: Good channel percentage per epoch ---
    ax2.bar(range(len(good_epochs_percentage)), good_epochs_percentage, 
            color=['red' if p else 'green' for p in problematic_epochs],
            alpha=0.7)
    ax2.axhline(threshold * 100, color='red', linestyle='--', linewidth=2, 
                label=f'{threshold*100:.0f}% threshold')
    ax2.set_xlabel('Epoch Number')
    ax2.set_ylabel('Good Channels (%)')
    ax2.set_ylim([0, 105])
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    # Add summary statistics
    n_bad = np.sum(rejection_log.bad_epochs)
    n_total = len(rejection_log.bad_epochs)
    n_problematic = np.sum(problematic_epochs)
    
    summary_text = (
        f'Marked Bad: {n_bad}/{n_total} ({n_bad/n_total:.1%})\n'
        f'Below {threshold*100:.0f}% threshold: {n_problematic} epochs'
    )
    
    ax1.text(
        0.02, 0.98, 
        summary_text,
        transform=ax1.transAxes,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
        fontsize=10
    )
    
    # Print problematic epochs to console (useful for debugging)
    if n_problematic > 0:
        print(f"\nProblematic epochs (< {threshold*100:.0f}% good channels) for subject {subject}:")
        for i in np.where(problematic_epochs)[0]:
            print(f"  Epoch {i}: {good_epochs_percentage[i]:.2f}% good channels")
    
    plt.tight_layout()
    return fig


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