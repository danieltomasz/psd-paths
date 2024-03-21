"""Utils for helping the anlysis of the data"""

import os
import sys
import configparser

config = configparser.ConfigParser()

# Create a ConfigParser object


def read_parameters(project_path: str):
    """ Get the parameters from the config.py file"""
    # Get the current working directory
    cwd = os.getcwd()

    # Go up directories until we reach the root of the project
    while os.path.basename(cwd) != 'root_of_your_project':
        cwd = os.path.dirname(cwd)

    # Add the root of the project to the Python path
    sys.path.append(cwd)

    # Read the config.py file
    config.read('config.py')

    return config