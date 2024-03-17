"""
File to run the pipeline.
Author: Daniel Borek, July 2023
"""

from pathlib import Path
import tomllib
import invoke
import tomllib


VARS = "PYDEVD_DISABLE_FILE_VALIDATION=1"
VENV = "conda-paths-3.12"
ENV = "miniforge3-latest"

PYTHON = f"~/.pyenv/versions/{ENV}/envs/{VENV}/bin/python"

with open(".env.toml", "rb") as toml_file:
    config = tomllib.load(toml_file)
project_path = config["project"]["path"]


#template = "notebooks/test/test0.ipynb"
template ="notebooks/1-preprocessing/template.ipynb"
output_dir = "notebooks/test"
output_dir = "notebooks/1-parametrized-reports"



@invoke.task(default=True)
def run_quarto_params(c):
    """Runs a quarto render."""
    subject = 102
    output_file = f"sub-{subject}.ipynb"
    c.run(
        f"quarto render {template} \\"
        f"-P subject:{subject}  --to ipynb  \\"
        f"--output {output_file} --output-dir {output_dir}"
    )
