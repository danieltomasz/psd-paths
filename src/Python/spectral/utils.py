"""Utils for helping the anlysis of the data"""

import os
import sys
from datetime import datetime
from pathlib import Path
import toml
from configparser import ConfigParser

from typing import List


def print_date_time():
    """ Print the current date and time"""
    now = datetime.now()
    # Convert the date and time to a string
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")

    # Print the date and time
    print(now_str)


def read_parameters(project_path: str = 'root_of_your_project',
                    settings_file: str = 'settings.toml'):
    """ Get the parameters from the config.py file"""
    cwd = os.getcwd()
    config = ConfigParser()

    # Iterate until we find the config.py file or the specified project folder
    while True:
        if os.path.exists(os.path.join(cwd, settings_file)):
            config_path = os.path.join(cwd, settings_file)
            config.read(config_path)
            break
        elif os.path.basename(cwd) == project_path:
            try:
                config_path = os.path.join(cwd, settings_file)
                config.read(config_path)
            except ImportError as exc:
                raise ImportError(f"Could not find {settings_file} in {cwd}") from exc
            else:
                break
        else:
            cwd = os.path.dirname(cwd)
            if cwd == '/':  # Reached the root directory
                raise FileNotFoundError(f"Could not find '{settings_file}'")
    return config


def config_project(subject: str | int, templates: dict | None = None) -> dict:
    """ Get the paths for given subject"""
    if templates is None:
        templates = {
            'project_path' : '{project_path}',
            'figures_path' : '{project_path}/analysis/figures/sub-{subject}/',
            'raw_preproc_path' : '{project_path}/analysis/data/raw/sub-{subject}/',
            'epochs_preproc_path' : '{project_path}/analysis/data/epochs/sub-{subject}/',
            'specparam_path' : '{project_path}/analysis/data/specparam/',
            'epochs_eeglab' : '{project_path}/analysis/data/eeglab/',
        }

    config = read_parameters()
    project_path = config.get("paths", "PROJECT_PATH").strip("'").strip('"')
    paths_dict = {}
    for var_name, var_value in templates.items():
        paths_dict[var_name] = var_value.format(project_path=project_path,
                                                subject=subject)
    return paths_dict

