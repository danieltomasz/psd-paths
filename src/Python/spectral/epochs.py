import mne

def create_epochs(raw: mne.io.Raw, length: float=5, overlap: float=0):
    """Create epochs from raw data"""
    events = mne.make_fixed_length_events(
        raw=raw,
        start=0,
        stop=None,
        duration=length,
        overlap=overlap,
        first_samp=False,
    )
    return mne.Epochs(
        raw=raw,
        events=events,
        tmin=0,
        tmax=length,
        detrend=1,  # from 0 to 1
        baseline=None,
        preload=True
    )
