"""
File to run the pipeline.
Author: Daniel Borek, July 2023
"""

from pathlib import Path
import tomllib
import re
import invoke


VARS = "PYDEVD_DISABLE_FILE_VALIDATION=1"
VENV = "conda-paths-3.12"
ENV = "miniforge3-latest"

PYTHON = f"~/.pyenv/versions/{ENV}/envs/{VENV}/bin/python"

with open(".env.toml", "rb") as toml_file:
    config = tomllib.load(toml_file)
project_path = config["project"]["path"]


#template = "notebooks/test/test0.ipynb"
#output_dir = "notebooks/test"

def extract_numbers(bids_paths):
    # Initialize an empty list to store the numbers
    numbers = []

    # Get all the child directories
    for path in bids_paths.glob("*"):
        if path.is_dir():
            # Convert the path to a string
            str_path = str(path)

            # Find all three-digit numbers in the string
            matches = re.findall(r"\b\d{3}\b", str_path)

            # Add the numbers to the list
            numbers.extend(matches)

    # Now 'numbers' is a list of all three-digit numbers in the child directory names
    return sorted(numbers)


def run_parametrised_report(c, subject, output_dir, report, output_file):
    """Basic function to run a parametrised report."""
    try:
        if not (Path(output_dir) / output_file).exists():
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            c.run(
                f"quarto render {report} \\"
                f"-P subject:{subject}  --to ipynb  \\"
                f"--output {output_file} "
            )
    except Exception as e:
        print(f"Error processing sub-{subject}: {e}")

@invoke.task(default=True)
def run_quarto_params(c):
    """
    Run parametrised quarto reports.
    """
    # get subjects
    bids_folder = Path(f"{project_path}/data/raw/BIDS/")
    subjects = extract_numbers(bids_folder)

    # set templates and output files
    output_dir = "analysis"

    templates = ["template-1-raw-notch-v2.ipynb",
                 "template-2-ica-specparam-v2.ipynb"]

    # iterate over subjects
    for subject in subjects:
        output_files = [f"sub-{subject}_1-raw-notch-v2.ipynb",
                        f"sub-{subject}_2-ica-specparam-v2.ipynb",
                        ]
        for template, output_file in zip(templates, output_files):
            print(f"Processing sub-{subject}")
            report = f"notebooks/1-preprocessing/{template}"
            run_parametrised_report(c,
                                    subject, output_dir, report, output_file)



@invoke.task()
def run_specparam(c):
    """
    Run parametrised quarto reports.
    """
    # get subjects
    bids_folder = Path(f"{project_path}/data/raw/BIDS/")
    subjects = extract_numbers(bids_folder)

    # set templates and output files
    output_dir = "analysis/last_stage"

    template =  "template-4-specparam.ipynb"

    # iterate over subjects
    for subject in subjects:
        output_file = f"sub-{subject}_4-specparam.ipynb"
        print(f"Processing sub-{subject}")
        report = f"notebooks/1-preprocessing/{template}"
        run_parametrised_report(c,
                                subject, output_dir, report, output_file)


@invoke.task()
def pipeline_notch(c):
    """
    Run parametrised quarto reports.
    """
    # get subjects
    bids_folder = Path(f"{project_path}/data/raw/BIDS/")
    subjects = extract_numbers(bids_folder)

    # set templates and output files
    output_dir = "analysis"

    template =  "template-1-raw-notch-v2.ipynb"

    # iterate over subjects
    for subject in subjects:
        output_file = f"sub-{subject}_1-raw-notch-v2.ipynb"
        print(f"Processing sub-{subject}")
        report = f"notebooks/1-preprocessing/{template}"
        run_parametrised_report(c,
                                subject, output_dir, report, output_file)