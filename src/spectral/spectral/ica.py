from mne_icalabel import label_components
from mne.preprocessing import ICA


def compute_ica(
    eeg_data,
    reject_log=None,
    n_components=0.999,
    method="picard",
    random_state: int = 99,
):
    """Compute ICA on the data without really bad epochs"""
    ica = ICA(
        n_components=n_components,
        random_state=random_state,
        method="picard",
        fit_params=dict(ortho=False, extended=True),
    )
    ica.fit(eeg_data)

    return ica


def label_components_ica(eeg_data, ica):
    """Assign the IC labels"""

    ic_labels = label_components(eeg_data, ica, method="iclabel")
    labels = ic_labels["labels"]
    return labels, ic_labels


def get_values(labels, ic_labels, threshold=0.8):
    """Get the indices of the labels"""
    element_indices = {}
    element_indices["bad_prob_class"] = []
    prababilities = ic_labels["y_pred_proba"]

    for i, element in enumerate(labels):
        if prababilities[i] > 0.8:
            if element not in element_indices:
                element_indices[element] = []
            element_indices[element].append(i)
        else:
            element_indices["bad_prob_class"].append(i)

    # display(element_indices)
    return element_indices


def plot_ica_components(ica, eeg_data, subject, figures_path, plot_properties=[]):
    """Plot the ICA components"""

    labels, ic_labels = label_components_ica(eeg_data, ica)
    labeled_components = get_values(labels, ic_labels)

    for label, indices in labeled_components.items():
        # display(label, indices)
        # ica_plot = ica.plot_sources(
        #    eeg_data, show_scrollbars=False, picks=slice(0, 20))
        print(label, indices)
        ica_plot_components = ica.plot_components(picks=indices)

        chunk_size = 20
        chunks = [
            indices[i : i + chunk_size] for i in range(0, len(indices), chunk_size)
        ]
        for chunk in chunks:
            ica_plot = ica.plot_sources(
                eeg_data,
                show_scrollbars=False,
                picks=chunk,
                start=0,
                stop=len(eeg_data) - 1,
            )

        if plot_properties:
            ica.plot_properties(eeg_data, picks=indices, psd_args={"fmax": 100.0})
    return labeled_components


def plot_removed_components(
    ica,
    eeg_data,
    figure_path,
    subject,
    labeled_components,
    chosen_components=[
        "eye blink",
        "heart beat",
        "line noise",
        "muscle artifact",
        "channel noise",
    ],
):
    """Plot the components labeled as eye blink, heart beat, line noise, muscle artifact, channel noise"""
    exclude = []
    for label, indices in labeled_components.items():
        if label in chosen_components:
            ica_plot = ica.plot_sources(eeg_data, show_scrollbars=False, picks=indices)
            print(label, indices)
            figs = ica.plot_properties(
                eeg_data, picks=indices, psd_args={"fmax": 100.0}
            )

            for fig, ind in zip(figs, indices):
                # fig = ica.plot_properties(eeg_data, picks=ind)
                fig.savefig(
                    f"{figure_path}/sub-{subject}_ICA{str(ind).zfill(3)}_{
                        label.replace(" ", "-")}.png",
                    dpi=300,
                    bbox_inches="tight",
                )
            exclude = exclude + indices
    return exclude
