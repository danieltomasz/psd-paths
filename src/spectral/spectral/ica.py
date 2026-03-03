from mne_icalabel import label_components
from mne.preprocessing import ICA
import mne
import numpy as np
from mne.preprocessing import ICA
import os
import pandas as pd

# Import display from IPython for notebook environments
try:
    from IPython.display import display
except ImportError:
    # Fallback for non-notebook environments
    def display(obj):
        print(obj)

def ica_params(config: dict) -> dict:
    """Extract ICA parameters from config"""
    # Check if ica_params is provided directly in config
    if "ica_params" in config:
        return config["ica_params"]
    
    # Otherwise, load from config["ica"] section
    return config.get("ica", {
        "n_components": 0.99,
        "method": "picard",
        "max_components": 50,
        "random_state": 99
    })

def compute_ica(
    eeg_data,
    reject_log=None,
    n_components=0.999,
    method="picard",
    random_state: int = 99
):
    """Compute ICA on the data"""
    ica = ICA(
        n_components=n_components,
        random_state=random_state,
        method=method,
        fit_params=dict(ortho=False, extended=True),
    )
    ica.fit(eeg_data)

    return ica


def label_components_ica(eeg_data, ica):
    """Assign the IC labels using ICLabel"""
    ic_labels = label_components(eeg_data, ica, method="iclabel")
    labels = ic_labels["labels"]
    return labels, ic_labels


def get_labeled_components(labels, ic_labels, threshold=0.8):
    """Get the indices of components by label with probability threshold"""
    element_indices = {}
    element_indices["bad_prob_class"] = []
    probabilities = ic_labels["y_pred_proba"]

    for i, element in enumerate(labels):
        if probabilities[i] > threshold:
            if element not in element_indices:
                element_indices[element] = []
            element_indices[element].append(i)
        else:
            element_indices["bad_prob_class"].append(i)

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
                        label.replace(' ', '-')
                    }.png",
                    dpi=300,
                    bbox_inches="tight",
                )
            exclude = exclude + indices
    return exclude



# GLOBAL SETTING: Keep this as 'warn' so the rest of the pipeline 
# doesn't crash on minor floating point issues.
np.seterr(all='warn')

def compute_ica_limited(eeg_data, variance_threshold=0.99, max_cap=40, random_state=97):
    """
    Computes ICA with 99% variance, capped at max_cap.
    Safely handles Picard failure and falls back to Infomax.
    """
    # 1. Calculate the rank integer
    ranks = mne.compute_rank(eeg_data)
    actual_rank = ranks['eeg'] if 'eeg' in ranks else sum(ranks.values())
    print(f"Data rank detected: {actual_rank}")
    
    # 2. Estimate components for variance threshold using FastICA
    # Wrapped in errstate to prevent notebook-level crashes during estimation
    print(f"Estimating component count for {variance_threshold*100}% variance...")
    try:
        with np.errstate(all='ignore'):
            tmp_ica = ICA(
                n_components=variance_threshold,
                method='fastica',
                random_state=random_state
            )
            tmp_ica.fit(eeg_data)
            n_variance = tmp_ica.n_components_
    except RuntimeError as e:
        # One component dominates variance (threshold too permissive for this data).
        # Fall back to using the full rank as the component estimate.
        print(f"Variance threshold estimation failed ({e}). Using full rank as estimate.")
        n_variance = actual_rank
    
    # 3. Apply the cap
    n_final = max(2, min(n_variance, max_cap, actual_rank - 1))

    print(f"Variance threshold requested {n_variance} components.")
    print(f"Applying cap: Final ICA will use {n_final} components.")

    # 4. Attempt Picard
    try:
        print(f"Attempting Picard...")
        ica = ICA(
            n_components=n_final,
            method='picard',
            random_state=random_state,
            fit_params=dict(ortho=False, extended=True),
            max_iter=500
        )
        
        # LOCAL OVERRIDE: We raise errors ONLY here to trigger the 'except' block
        # if Picard hits a numerical stability issue.
        with np.errstate(all='raise'):
            ica.fit(eeg_data)
        print("Success: Picard converged.")

    # 5. Fallback to Infomax
    except (FloatingPointError, RuntimeWarning, ValueError, Exception) as e:
        print(f"Picard failed ({type(e).__name__}: {e}). Falling back to Infomax...")
        ica = ICA(
            n_components=n_final,
            method='infomax',
            random_state=random_state,
            fit_params=dict(extended=True),
            max_iter='auto'
        )
        # We go back to 'ignore' or 'warn' for the fallback to ensure it finishes
        with np.errstate(all='warn'):
            ica.fit(eeg_data)
        print("Success: Infomax converged.")
        
    return ica

from mne_icalabel import label_components

def plot_annotated_topographies(ica, eeg_data, ic_labels, report=None):
    """
    Plots all ICA component topographies annotated with ICLabel names and probabilities.
    """
    labels = ic_labels["labels"]
    probs = ic_labels["y_pred_proba"]
    
    # Create titles mapping IC index to "Label (Probability%)"
    titles = {
        i: f"IC{i:02d}: {label} ({probs[i]*100:.1f}%)" 
        for i, label in enumerate(labels)
    }

    # Generate topographies. Use inst=eeg_data to ensure proper spatial context.
    figs = ica.plot_components(inst=eeg_data, title="ICA Component Classification", show=False)
    
    if not isinstance(figs, list):
        figs = [figs]

    ic_idx = 0
    for i, fig in enumerate(figs):
        # Apply custom titles to each subplot axis
        for ax in fig.axes:
            if ic_idx < len(labels):
                ax.set_title(titles[ic_idx], fontsize=9)
                ic_idx += 1
        
        # Display in notebook and add to MNE Report
        display(fig)
        if report is not None:
            report.add_figure(fig, title=f"ICA Topographies - Page {i+1}")

    return figs

def plot_component_properties(ica, eeg_data, picks, ic_labels=None, report=None, save_path=None, subject=""):
    """
    Plots detailed properties for a specific list of component indices.
    """
    if not picks:
        print("No components provided for property plotting.")
        return []

    # Get labels for titles if provided
    titles = {}
    if ic_labels:
        for i in picks:
            label = ic_labels["labels"][i]
            prob = ic_labels["y_pred_proba"][i]
            titles[i] = f"IC {i:02d}: {label} ({prob*100:.1f}%)"
    else:
        titles = {i: f"IC {i:02d}" for i in picks}

    # Generate property plots
    figs = ica.plot_properties(eeg_data, picks=picks, show=False, psd_args={"fmax": 100.0})

    for i, fig in zip(picks, figs):
        fig.suptitle(titles[i], fontsize=12)
        display(fig)
        
        # Save to disk if path is provided
        if save_path:
            label_slug = titles[i].split(":")[1].split("(")[0].strip().replace(" ", "-")
            filename = f"sub-{subject}_IC{i:02d}_{label_slug}.png"
            fig.savefig(f"{save_path}/{filename}", dpi=300, bbox_inches="tight")

        # Add to MNE Report
        if report is not None:
            report.add_figure(fig, title=f"Properties {titles[i]}", tags=("ica", "artifact"))

    return figs

def get_manual_ica_list(subject_id,  log_path):
    """
    Checks the CSV for manual overrides. 
    Returns a list of indices if found, else returns an empty list.
    """
    if os.path.exists(log_path):
        df_log = pd.read_csv(log_path)
        # Match subject_id as a string
        sub_row = df_log[df_log['subject_id'].astype(str) == str(subject_id)]
        
        if not sub_row.empty:
            manual_val = sub_row.iloc[0]['manual_exclude']
            
            # --- ROBUST CHECK FOR NAN AND EMPTY STRINGS ---
            if pd.isna(manual_val) or str(manual_val).strip().lower() == 'nan' or str(manual_val).strip() == '':
                return [] # Fall back to automated detection
            # ----------------------------------------------

            print(f"✅ Using manual override from CSV for sub-{subject_id}")
            # Convert "0, 1, 2" string to [0, 1, 2]
            return [int(x.strip()) for x in str(manual_val).split(',') if x.strip().isdigit()]
            
    return []
