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

from run_pipeline import STEPS  # same steps as the default run
from spectral.runner import PipelineConfig, run_pipeline, show_status

cfg = PipelineConfig.from_settings(
    steps=STEPS,
    derivatives_root="_test/derrivatives",
    outputs_root="_test/outputs",
)


if __name__ == "__main__":
    print(cfg.describe())
    run_pipeline(cfg, n_subjects=8, skip_completed=False)
    show_status(cfg)
