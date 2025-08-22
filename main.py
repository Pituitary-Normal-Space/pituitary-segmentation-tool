"""
Main file that can be called to run the project. We may provide a UI or command line interface to run the project as well.
"""

# Standard library imports
import multiprocessing as mp

# Third party imports
# import fslpy as fsl
import pandas as pd

# Local application imports
from utils import create_subject_list
from config import config
from const import PATH_TO_UNPROCESSED_DATA, KEY_MAP_NAME


if config.percent_cpus_to_use > 1:
    raise ValueError("percent_cpus_to_use must be a float between 0 and 1")

if config.percent_cpus_to_use <= 0:
    raise ValueError("percent_cpus_to_use must be greater than 0")

if type(config.percent_cpus_to_use) is not float:
    raise ValueError("percent_cpus_to_use must be a float between 0 and 1")


def process_subject(subject):
    """Process a single subject"""
    print(subject)
    # subject.preprocess_MRIs()
    # subject.overlay_MRIs()
    # subject.coregister_to_mni_space()
    subject.setup_pituitary_analysis()
    subject.segment_pituitary_gland(weighted_img_to_use="both")
    print(subject)


def main() -> None:
    # Get the list of subject objects
    subject_list = create_subject_list(
        pd.read_csv(f"{PATH_TO_UNPROCESSED_DATA}/{KEY_MAP_NAME}")
    )

    # Make subject list just a single subject
    subject_list = subject_list[:1]

    # Determine number of cores to use based on percent_cpus_to_use
    num_cores = max(1, int(mp.cpu_count() * config.percent_cpus_to_use))
    print(f"Using {num_cores} cores for processing")

    # Create a pool of workers
    with mp.Pool(processes=num_cores) as pool:
        # Map the process_subject function to all subjects
        pool.map(process_subject, subject_list)


if __name__ == "__main__":
    main()
