"""
Tests new configuration parameters from the tuning process.
"""

# Standard library imports
import os

# Third party imports
import pandas as pd

# Internal imports
from config import config
from subject import Subject
from utils import create_subject_list, get_DICE
from const import PATH_TO_GROUND_TRUTH_MASKS, KEY_MAP_NAME, PATH_TO_UNPROCESSED_DATA


def run_segmentation(subject: Subject) -> str:
    """
    Run the segmentation pipeline for a given subject.

    :param subject: Subject object containing MRI data and configuration.
    :return: string path to the predicted mask.
    """
    subject.setup_pituitary_analysis()
    subject.segment_pituitary_gland()
    return subject.binary_pituitary_mask


def test_config(score_based=True) -> None:
    """
    Test the configuration parameters by running the pipeline with the current configuration.
    """
    if score_based:
        # Update the configuration
        centroid_x = 1
        centroid_y = 3
        centroid_z = -35
        config.default_centroid = (centroid_x, centroid_y, centroid_z)
        x_range_min = -11
        x_range_max = 14
        config.x_range = (x_range_min, x_range_max)
        y_range_min = -3
        y_range_max = 8
        config.y_range = (y_range_min, y_range_max)
        z_range_min = -43
        z_range_max = -29
        config.z_range = (z_range_min, z_range_max)
        intensity_range_min = 471
        intensity_range_max = 931
        config.intensity_range = (intensity_range_min, intensity_range_max)
        config.max_voxel_drift = 2
        config.distance_weight = 0.5176632767548813
        config.intensity_range_weight = 0.36796023144059437
        config.connectivity_weight = 1 - (
            config.distance_weight + config.intensity_range_weight
        )
        config.high_quality_neighbors_to_consider_connected = 10
        config.min_score_threshold = 0.6560892197893082
        config.num_neighbors_required_to_boost = 7
        config.min_score_to_boost_if_quality_neighbors = 0.6963053558332413
        config.min_score_considered_high_score = 0.6720906241165322
        config.do_appendage_removal = True
        config.infundibulum_range = 5
        config.appendage_removal_radius = 5
        config.final_score_threshold = 0.31931196036358167

    # Create subject list
    subject_list = create_subject_list(
        pd.read_csv(f"{PATH_TO_UNPROCESSED_DATA}/{KEY_MAP_NAME}")
    )

    # Only leave subjects from subject list that a subject_id that matches a ground truth mask
    subject_list = [
        subject
        for subject in subject_list
        if any(
            subject.subject_id in file
            for file in os.listdir(PATH_TO_GROUND_TRUTH_MASKS)
        )
    ]

    # Get last two subjects
    subject_list = subject_list[-2:]

    print("Selected subjects:", [subject.subject_id for subject in subject_list])

    # Run segmentation for each subject and collect predicted masks
    predicted_masks = {
        subject.subject_id: run_segmentation(subject) for subject in subject_list
    }  # Replace with your segmentation function

    # Compute DICE scores
    DICE_scores = []
    for subject_id, predicted_mask in predicted_masks.items():
        ground_truth_mask_path = os.path.join(
            PATH_TO_GROUND_TRUTH_MASKS,
            f"{subject_id}.nii.gz",
        )
        # Calculate DICE score for each subject
        print("Calculating DICE score for subject", subject_id)
        # t1 = [subj.t1_in_mni_space for subj in subject_list if subj.subject_id == subject_id][0]
        DICE_scores.append(
            get_DICE(
                ground_truth_mask_path,
                predicted_mask,
            )
        )

    print("DICE scores:", DICE_scores)

    # Calculate the average DICE score
    average_DICE_score = sum(DICE_scores) / len(DICE_scores)
    print("Average DICE score:", average_DICE_score)


test_config()
