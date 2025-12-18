# mne_pipeline.py
import mne
import numpy as np
import matplotlib.pyplot as plt
from hamilton.function_modifiers import parameterize, tag, source, value
from spectral.utils import ProjectPaths, load_config
from spectral.helpers import load_or_create_report
from spectral.ica import (
    compute_ica,
    label_components_ica,
    get_labeled_components
)


# --- NODE 1: Configuration & Paths ---
def config() -> dict:
    """Loads settings.toml"""
    return load_config()


def paths(subject_id: str) -> ProjectPaths:
    """Creates project paths object"""
    paths = ProjectPaths(subject_id)
    paths.create_directories()
    return paths



@tag(kind="report")
def report_loaded(
    paths: ProjectPaths, 
    subject_id: str,
    append_to_existing: bool = False  # Start fresh for preprocessing
) -> mne.Report:
    """Loads or creates report for preprocessing"""
    return load_or_create_report(
        paths, 
        subject_id, 
        append_to_existing,
        stage="Preprocessing"
    )

def epochs_loaded(subject_id: str, paths: ProjectPaths) -> mne.Epochs:
    """Loads saved good epochs from disk"""
    epochs_path = paths.epochs / f"sub-{subject_id}_good_epochs-epo.fif"
    
    if not epochs_path.exists():
        raise FileNotFoundError(f"Could not find epochs file: {epochs_path}")
    
    return mne.read_epochs(epochs_path, preload=True)

def ica_computed(
    epochs_loaded: mne.Epochs,
    ica_params: dict
) -> mne.preprocessing.ICA:
    """Computes ICA on clean epochs"""
    n_components = ica_params.get("n_components", 0.99)
    method = ica_params.get("method", "picard")
    max_components = ica_params.get("max_components", 50)
    random_state = ica_params.get("random_state", 99)
    
    ica = compute_ica(
        epochs_loaded, 
        n_components=n_components, 
        method=method,
        random_state=random_state
    )
    print(f"ICA computed: {ica.n_components_} components")
    
    if ica.n_components_ > max_components:
        print(f"Recomputing with {max_components} components")
        ica = compute_ica(
            epochs_loaded, 
            n_components=max_components, 
            method=method,
            random_state=random_state
        )
    
    return ica


def ica_labels(
    epochs_loaded: mne.Epochs,
    ica_computed: mne.preprocessing.ICA
) -> dict:
    """Labels ICA components using ICLabel"""
    labels, ic_labels = label_components_ica(epochs_loaded, ica_computed)
    print(f"ICLabel classification complete")
    return {
        "labels": labels,
        "ic_labels": ic_labels
    }



def labeled_components(
    ica_labels: dict,  # Changed from tuple to dict
    threshold: float = 0.8
) -> dict:
    """Groups components by label with probability threshold"""
    labels = ica_labels["labels"]  # Extract from dict
    ic_labels = ica_labels["ic_labels"]  # Extract from dict
    
    components_dict = get_labeled_components(labels, ic_labels, threshold)
    
    # Print summary
    print("\nLabeled components summary:")
    for label, indices in components_dict.items():
        print(f"  {label}: {indices}")
    
    return components_dict


def artifact_components(
    labeled_components: dict,
    artifact_types: list[str] = None
) -> list[int]:
    """Extract artifact component indices for removal"""
    if artifact_types is None:
        artifact_types = [
            "eye blink",
            "heart beat",
            "line noise",
            "muscle artifact",
            "channel noise"
        ]
    
    exclude = []
    for label, indices in labeled_components.items():
        if label in artifact_types:
            exclude.extend(indices)
    
    exclude = sorted(exclude)
    print(f"\nArtifact components to exclude: {exclude}")
    return exclude


@tag(kind="visualization")
def plot_ica_components_all(
    ica_computed: mne.preprocessing.ICA,
    subject_id: str
) -> plt.Figure:
    """Plots all ICA components topography"""
    fig = ica_computed.plot_components(show=False)
    return fig


@tag(kind="visualization")
def plot_ica_components_by_label(
    ica_computed: mne.preprocessing.ICA,
    labeled_components: dict,
    subject_id: str
) -> dict[str, plt.Figure]:
    """Plots ICA components grouped by ICLabel classification"""
    figures = {}
    
    for label, indices in labeled_components.items():
        if indices:  # Only plot if there are components
            fig = ica_computed.plot_components(picks=indices, show=False)
            figures[label] = fig
            plt.close(fig)
    
    return figures


@tag(kind="visualization")
def plot_artifact_sources(
    ica_computed: mne.preprocessing.ICA,
    epochs_loaded: mne.Epochs,
    artifact_components: list[int],
    subject_id: str,
    chunk_size: int = 20
) -> list[plt.Figure]:
    """Plots time courses of artifact components"""
    if not artifact_components:
        return []
    
    figures = []
    chunks = [
        artifact_components[i:i + chunk_size] 
        for i in range(0, len(artifact_components), chunk_size)
    ]
    
    for chunk in chunks:
        fig = ica_computed.plot_sources(
            epochs_loaded,
            show_scrollbars=False,
            picks=chunk,
            start=0,
            stop=len(epochs_loaded) - 1,
            show=False
        )
        figures.append(fig)
        plt.close(fig)
    
    return figures


@tag(kind="visualization")
def plot_artifact_properties(
    ica_computed: mne.preprocessing.ICA,
    epochs_loaded: mne.Epochs,
    artifact_components: list[int],
    labeled_components: dict,
    subject_id: str,
    fmax: float = 100.0
) -> dict[str, list[plt.Figure]]:
    """Plots detailed properties of artifact components grouped by type"""
    artifact_types = [
        "eye blink", "heart beat", "line noise", 
        "muscle artifact", "channel noise"
    ]
    
    figures_by_type = {}
    
    for label, indices in labeled_components.items():
        if label in artifact_types and indices:
            figs = ica_computed.plot_properties(
                epochs_loaded,
                picks=indices,
                psd_args={"fmax": fmax},
                show=False
            )
            figures_by_type[label] = figs
            
            for fig in figs:
                plt.close(fig)
    
    return figures_by_type


def saved_ica_path(
    ica_computed: mne.preprocessing.ICA,
    paths: ProjectPaths,
    subject_id: str
) -> str:
    """Saves ICA model to disk"""
    out_path = paths.ica / f"sub-{subject_id}_my_ica_model-ica.fif"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    ica_computed.save(out_path, overwrite=True)
    print(f"ICA saved to {out_path}")
    return str(out_path)


@tag(kind="report")
def report_with_ica(
    report_loaded: mne.Report,  # From shared report module
    plot_ica_components_all: plt.Figure,
    plot_ica_components_by_label: dict[str, plt.Figure],
    plot_artifact_properties: dict[str, list[plt.Figure]],
    labeled_components: dict,
    artifact_components: list[int]
) -> mne.Report:
    """Adds ICA analysis to the report"""
    
    # 1. All components
    report_loaded.add_figure(
        fig=plot_ica_components_all,
        title="ICA - All Components",
        caption="Topographic maps of all ICA components.",
        tags=("ica", "components", "all")
    )
    
    # 2. Components by label
    for label, fig in plot_ica_components_by_label.items():
        report_loaded.add_figure(
            fig=fig,
            title=f"ICA - {label.title()} Components",
            caption=f"Components classified as '{label}' by ICLabel.",
            tags=("ica", "components", label.replace(" ", "_"))
        )
    
    # 3. Artifact properties (detailed)
    for label, figs in plot_artifact_properties.items():
        for i, fig in enumerate(figs):
            comp_idx = labeled_components[label][i]
            report_loaded.add_figure(
                fig=fig,
                title=f"ICA Component {comp_idx} - {label.title()}",
                caption=f"Detailed properties of component {comp_idx} classified as '{label}'.",
                tags=("ica", "properties", label.replace(" ", "_"), f"comp{comp_idx}")
            )
    
    # 4. Summary text
    summary = f"Total components: {sum(len(v) for v in labeled_components.values())}\n"
    summary += f"Artifacts to remove: {len(artifact_components)}\n\n"
    summary += "Components by type:\n"
    for label, indices in labeled_components.items():
        summary += f"  {label}: {len(indices)} {indices}\n"
    
    print(summary)
    
    return report_loaded