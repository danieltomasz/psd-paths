"""December-style pipeline run.

Same runner, different templates and different output locations, so it cannot
collide with the default run's derivatives, notebooks or status file.

    uv run python templates/run_december.py
"""

from spectral.runner import PipelineConfig, Step, run_pipeline, show_status

DECEMBER_STEPS = {
    1: Step("templates/december/step1-preprocessing.ipynb", "step1-preprocessing", "Preprocessing"),
    2: Step("templates/december/step1b-epochs-psd-analysis.ipynb", "step1b-epochs", "Epochs"),
    3: Step("templates/december/step2-psd-analysis.ipynb", "step2-psd-analysis", "ICA+Specparam"),
}

cfg = PipelineConfig.from_settings(
    steps=DECEMBER_STEPS,
    output_root="outputs/pipeline_december",
    status_file="outputs/pipeline_status_december.json",
)


if __name__ == "__main__":
    run_pipeline(cfg, n_subjects=10, steps_to_run=[1], skip_completed=False)
    show_status(cfg)
