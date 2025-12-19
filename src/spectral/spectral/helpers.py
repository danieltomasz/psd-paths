import mne
from spectral.utils import ProjectPaths


def load_or_create_report(
    paths: ProjectPaths, 
    subject_id: str,
    append_to_existing: bool = True,
    stage: str = "Analysis"
) -> mne.Report:
    """
    Loads existing H5 report or creates new one.
    Set append_to_existing=False to start from scratch.
    """
    h5_path = paths.reports / f"sub-{subject_id}_report.h5"
    
    # Start from scratch if requested
    if not append_to_existing:
        print(f"Creating new report from scratch for subject {subject_id}")
        return mne.Report(
            title=f"Subject {subject_id} {stage} Report", 
            verbose=False
        )
    
    # Try to load existing report
    if h5_path.exists():
        print(f"Loading existing report from {h5_path}")
        return mne.open_report(h5_path)
    else:
        print(f"Report file not found at {h5_path}, creating new report")
        return mne.Report(
            title=f"Subject {subject_id} Analysis Report", 
            verbose=False
        )
    
def style_cfg(*, node, node_class): 
    """
    Returns: (style_dict, node_class, legend_label)
    """
    tag_value = node.tags.get("kind")

    # Style 1: Visualization (Gold)
    if tag_value == "visualization":
        return ({"style": "filled", "fillcolor": "#FFD700"}, node_class, "Plotting")
    
    # Style 2: Report (Pale Green) <--- NEW SECTION
    elif tag_value == "report":
        return ({"style": "filled", "fillcolor": "#98FB98"}, node_class, "Report")
    
    # Default: Data Processing (Blue/Grey default)
    return ({}, node_class, None)