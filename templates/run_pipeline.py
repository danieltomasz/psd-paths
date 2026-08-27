"""Default PATHS pipeline run.

All logic lives in `spectral.runner`. This file only chooses *what* to run.
To make a pipeline variant, copy this file and change the config -- see
`run_december.py` for an example.

    uv run python templates/run_pipeline.py

    # or interactively
    from spectral.runner import PipelineConfig, run_pipeline, show_status
    cfg = PipelineConfig.from_settings()
    run_pipeline(cfg, n_subjects=10, steps_to_run=[1])
"""

from spectral.runner import PipelineConfig, rerun_failed, run_pipeline, show_status  # noqa: F401

# Templates, output locations, n_jobs and kernel all come from settings.toml.
cfg = PipelineConfig.from_settings()


if __name__ == "__main__":
    # First 10 subjects, preprocessing only.
    run_pipeline(cfg, n_subjects=10, steps_to_run=[1], skip_completed=False)
    show_status(cfg)

    # --- other usage ---
    # run_pipeline(cfg)                                          # everything
    # run_pipeline(cfg, subjects_to_run=["101", "127"], steps_to_run=[2, 3])
    # run_pipeline(cfg, subject_steps_map={"101": [2, 3], "127": [3]})
    # rerun_failed(cfg)
