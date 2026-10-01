"""Default PATHS pipeline run.

All logic lives in `spectral.runner`. This file only chooses *what* to run.

    uv run python templates/run_pipeline.py

    # or interactively
    from run_pipeline import cfg, run_group
    from spectral.runner import run_pipeline, show_status
    run_pipeline(cfg, n_subjects=10, steps_to_run=[1])

    # a test run into its own folder, settings.toml unchanged
    test = PipelineConfig.from_settings(steps=STEPS,
        derivatives_root="runs/_test/derivatives", outputs_root="runs/_test/outputs")
"""

from pathlib import Path

import papermill as pm

from spectral.runner import PipelineConfig, Step, rerun_failed, run_pipeline, show_status  # noqa: F401

# Steps run in this order, once per subject. Output notebooks are named after the
# template (sub-XXX_04_ICA_apply_interpolate.ipynb, ...). eeg-spectral's own
# default list still has the older 6-step layout, so the steps are set here.
STEPS = [
    Step("01_Preprocessing.ipynb", name="Preprocessing"),
    Step("02_Epochs.ipynb", name="Epochs"),
    Step("03_ICA_fit.ipynb", name="ICA fit"),
    Step("04_ICA_apply_interpolate.ipynb", name="ICA apply and interpolate"),
    Step("05_Specparam.ipynb", name="Specparam"),
]

# Runs once per run, after all subjects: group tables and exclusion report.
GROUP_TEMPLATE = Path(__file__).parent / "SpecparamTogether.ipynb"

# Templates, output locations, n_jobs and kernel come from settings.toml.
cfg = PipelineConfig.from_settings(steps=STEPS)


def run_group(cfg):
    """Execute the group notebook once for the run that `cfg` describes."""
    out = cfg.output_root / GROUP_TEMPLATE.name
    out.parent.mkdir(parents=True, exist_ok=True)
    pm.execute_notebook(
        GROUP_TEMPLATE, out,
        parameters={"config_path": str(cfg.config_path) if cfg.config_path else None,
                    "path_overrides": cfg.path_overrides or None},
        kernel_name=cfg.kernel_name, progress_bar=False,
    )
    return out


if __name__ == "__main__":
    # All subjects, all steps, into the run folder set in settings.toml,
    # then the group tables.
    run_pipeline(cfg, skip_completed=False)
    show_status(cfg)
    print(f"group step: {run_group(cfg)}")

    # --- other usage ---
    # run_pipeline(cfg, n_subjects=10, steps_to_run=[1])        # quick test
    # run_pipeline(cfg, subjects_to_run=["101", "127"], steps_to_run=[2, 3])
    # run_pipeline(cfg, subject_steps_map={"101": [2, 3], "127": [3]})
    # rerun_failed(cfg)
