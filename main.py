"""
Main file that can be called to run the project. We may provide a UI or command line interface to run the project as well.
"""

# Third party imports
# import fslpy as fsl
import pandas as pd

# Local application imports
from utils import create_subject_list
from const import PATH_TO_UNPROCESSED_DATA, KEY_MAP_NAME


def main() -> None:
    # Get the list of subject objects
    subject_list = create_subject_list(
        pd.read_csv(f"{PATH_TO_UNPROCESSED_DATA}/{KEY_MAP_NAME}")
    )
    for subject in subject_list:
        if subject.subject_id != "101107":
            continue
        subject.setup_pituitary_analysis()
        print(subject)
        # subject.preprocess_MRIs()
        # subject.overlay_MRIs()
        # subject.coregister_to_mni_space()
        subject.segment_pituitary_gland()
        print(subject)


main()
