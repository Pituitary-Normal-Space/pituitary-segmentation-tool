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
from config import percent_cpus_to_use
from const import PATH_TO_UNPROCESSED_DATA, KEY_MAP_NAME


if percent_cpus_to_use > 1:
    raise ValueError("percent_cpus_to_use must be a float between 0 and 1")

if percent_cpus_to_use <= 0:
    raise ValueError("percent_cpus_to_use must be greater than 0")

if type(percent_cpus_to_use) is not float:
    raise ValueError("percent_cpus_to_use must be a float between 0 and 1")


def process_subject(subject):
    """Process a single subject"""
    if subject.subject_id in [
        "101107",
        "100307",
        "100408",
        "101309",
        "101915",
        "103111",
        "103414",
        "106016",
        "108828",
        "110411",
        "113922",
    ]:
        print(f"Skipping {subject.subject_id}")
        return

    print(subject)
    subject.preprocess_MRIs()
    subject.overlay_MRIs()
    subject.coregister_to_mni_space()
    subject.segment_pituitary_gland()
    print(subject)


def main() -> None:
    # Get the list of subject objects
    subject_list = create_subject_list(
        pd.read_csv(f"{PATH_TO_UNPROCESSED_DATA}/{KEY_MAP_NAME}")
    )

    # Determine number of cores to use based on percent_cpus_to_use
    num_cores = max(1, int(mp.cpu_count() * percent_cpus_to_use))
    print(f"Using {num_cores} cores for processing")

    # Create a pool of workers
    with mp.Pool(processes=num_cores) as pool:
        # Map the process_subject function to all subjects
        pool.map(process_subject, subject_list)


if __name__ == "__main__":
    main()
