"""Default PATHS pipeline run.

All logic lives in `spectral.runner`. This file only chooses *what* to run.
To make a pipeline variant, copy this file and change the config -- see
`run_december.py` for an example.

    uv run python templates/run_pipeline.py

    # or interactively
    from run_pipeline import cfg
    from spectral.runner import run_pipeline, show_status
    run_pipeline(cfg, n_subjects=10, steps_to_run=[1])
"""

from spectral.runner import PipelineConfig, Step, rerun_failed, run_pipeline, show_status  # noqa: F401

# Steps run in this order. Output notebooks are named after the template
# (sub-XXX_04_ICA_apply_interpolate.ipynb, ...). eeg-spectral's own default list
# still has the older 6-step layout, so the steps are set here.
STEPS = [
    Step("01_Preprocessing.ipynb", name="Preprocessing"),
    Step("02_Epochs.ipynb", name="Epochs"),
    Step("03_ICA_fit.ipynb", name="ICA fit"),
    Step("04_ICA_apply_interpolate.ipynb", name="ICA apply and interpolate"),
    Step("05_Specparam.ipynb", name="Specparam"),
]

# Templates, output locations, n_jobs and kernel come from settings.toml.
cfg = PipelineConfig.from_settings(steps=STEPS)


if __name__ == "__main__":
    # First 10 subjects, preprocessing only.
    run_pipeline(cfg, n_subjects=10, steps_to_run=[1], skip_completed=False)
    show_status(cfg)

    # --- other usage ---
    # run_pipeline(cfg)                                          # everything
    # run_pipeline(cfg, subjects_to_run=["101", "127"], steps_to_run=[2, 3])
    # run_pipeline(cfg, subject_steps_map={"101": [2, 3], "127": [3]})
    # rerun_failed(cfg)
