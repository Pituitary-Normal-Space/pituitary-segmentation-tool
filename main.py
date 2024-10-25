"""
Main file that can be called to run the project. We may provide a UI or command line interface to run the project as well.
"""

# Standard library imports
import os

# Third party imports
# import fslpy as fsl
import pandas as pd
import nibabel as nib

# Local application imports
from utils import create_subject_list
from const import PATH_TO_DATA, KEY_MAP_NAME


def main():
    # Get the list of subject objects
    subject_list = create_subject_list(pd.read_csv(f"{PATH_TO_DATA}/{KEY_MAP_NAME}"))
    _ = [print(subject) for subject in subject_list]


main()
