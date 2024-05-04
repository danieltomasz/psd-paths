import mne
import autoreject


def create_epochs(raw: mne.io.Raw, length: float = 5, overlap: float =1.5):
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
        preload=True,
        reject_by_annotation=True
    )


def get_reject_log(epochs, resample=None, consensus=[0.8],
                   n_interpolate=None):
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