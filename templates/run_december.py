"""Pipeline run writing everything into _test/.

    uv run python templates/run_december.py

WHERE THINGS GO
---------------
`settings.toml` stays untouched -- this script overrides two path knobs, and the
runner forwards them to every notebook so data and bookkeeping cannot drift apart:

    derivatives_root = "_test/derrivatives"   ->  the .fif files
    outputs_root     = "_test/outputs"        ->  figures, reports, specparam,
                                                  executed notebooks, run logs,
                                                  the status file

Input BIDS is not affected. Delete _test/ to throw the whole run away.
Run `print(cfg.describe())` to see the resolved plan before executing.
"""

from spectral.runner import PipelineConfig, Step, run_pipeline, show_status

# Steps run in list order. Only `template` is required -- `name` (used in logs,
# the status table, and report headings) and `suffix` (names the executed
# notebook) are derived from the filename unless given.
# Bare filenames resolve against templates_dir; give a subpath or an absolute
# path to pull a template from anywhere else.
STEPS = [
    Step(template="01_Preprocessing.ipynb", name="Preprocessing"),
    Step(template="02_Epochs.ipynb", name="Epoching"),
    Step(template="03_ICA_fit.ipynb", name="ICA"),
    Step(template="04_ICA_apply.ipynb", name="ICA applied"),
    Step(template="05_Interpolate.ipynb", name="Interpolate"),
    Step(template="06_Specparam.ipynb", name="Spectral parameterisation"),
]

cfg = PipelineConfig.from_settings(
    steps=STEPS,
    # templates_dir="templates/december",   # pull STEPS from a different folder
    derivatives_root="_test/derrivatives",
    outputs_root="_test/outputs",
)


if __name__ == "__main__":
    print(cfg.describe())
    # 1-6: the average-reference fix changed step 1's output, so everything
    # downstream of it is stale.
    run_pipeline(cfg, n_subjects=8, steps_to_run=[1, 2, 3, 4, 5, 6], skip_completed=False)
    show_status(cfg)
