# Built in imports
import os

# Internal imports
from const import PATH_TO_GROUND_TRUTH_MASKS, KEY_MAP_NAME, PATH_TO_UNPROCESSED_DATA
from config import config

# External imports
import optuna
import numpy as np
import pandas as pd


def run_segmentation(subject) -> str:
    """
    Run the segmentation pipeline for a given subject.

    :param subject: Subject object containing MRI data and configuration.
    :return: string path to the predicted mask.
    """
    subject.setup_pituitary_analysis()
    subject.segment_pituitary_gland()
    return subject.binary_pituitary_mask


def objective(trial) -> float:
    """
    Objective function for Optuna optimization.
    :param trial: Optuna trial object.

    :return: Weighted average of DICE scores.
    """
    # Set range of hyperparameters to tune
    # x coordinate: search around 0 with ±5 range
    x = trial.suggest_int("centroid_x", -2, 2)

    # y coordinate: search around 2 with ±5 range
    y = trial.suggest_int("centroid_y", -1, 3)

    # z coordinate: search around -32 with ±5 range
    z = trial.suggest_int("centroid_z", -37, -30)

    # Update config with new centroid
    config.default_centroid = (x, y, z)

    # Now tune frame of reference
    minimum = trial.suggest_int("x_range_min", -12, -8)
    maximum = trial.suggest_int("x_range_max", 10, 14)
    config.x_range = (minimum, maximum)

    minimum = trial.suggest_int("y_range_min", -5, -1)
    maximum = trial.suggest_int("y_range_max", 8, 12)
    config.y_range = (minimum, maximum)

    minimum = trial.suggest_int("z_range_min", -43, -39)
    maximum = trial.suggest_int("z_range_max", -29, -25)
    config.z_range = (minimum, maximum)

    # Now tune intensity range
    minimum = trial.suggest_int("intensity_range_min", 100, 500)
    maximum = trial.suggest_int("intensity_range_max", 500, 1000)
    config.intensity_range = (minimum, maximum)

    # Tune voxel drift
    config.max_voxel_drift = trial.suggest_int("max_voxel_drift", 0, 3)

    # Tune score based weights - using only two parameters and calculating the third
    config.distance_weight = trial.suggest_float("distance_weight", 0, 1)
    config.intensity_range_weight = trial.suggest_float(
        "intensity_range_weight", 0, 1 - config.distance_weight
    )
    # Connectivity weight is automatically set to complement to 1
    config.connectivity_weight = 1 - (
        config.distance_weight + config.intensity_range_weight
    )

    # Tune "high quality" neighbors
    config.high_quality_neighbors_to_consider_connected = trial.suggest_int(
        "high_quality_neighbors_to_consider_connected", 2, 10
    )
    # Tune min score threshold
    config.min_score_threshold = trial.suggest_float("min_score_threshold", 0.5, 1)
    # Tune intensity tolerance
    config.intensity_tolerance = trial.suggest_int("intensity_tolerance", 10, 500)
    # Tune max voxels
    config.max_voxels = trial.suggest_int("max_voxels", 5000, 10000)
    # Tune region growing weight
    config.score_based_weight = trial.suggest_float("score_based_weight", 0, 1)
    config.region_growing_weight = (
        1 - config.score_based_weight
    )  # Automatically complements to 1

    config.num_neighbors_required_to_boost = trial.suggest_int(
        "num_neighbors_required_to_boost", 4, 26
    )
    config.min_score_to_boost_if_quality_neighbors = trial.suggest_float(
        "min_score_to_boost_if_quality_neighbors", 0, 1
    )
    config.min_score_considered_high_score = trial.suggest_float(
        "min_score_considered_high_score", 0.5, 1
    )
    # Tune appendage removal
    config.do_appendage_removal = trial.suggest_categorical(
        "do_appendage_removal", [True, False]
    )
    # Tune infundibulum range
    config.infundibulum_range = trial.suggest_int("infundibulum_range", 0, 10)
    # Tune appendage removal radius
    config.appendage_removal_radius = trial.suggest_int(
        "appendage_removal_radius", 1, 5
    )
    # Tune final score threshold
    config.final_score_threshold = trial.suggest_float("final_score_threshold", 0.3, 1)

    from utils import create_subject_list, get_DICE

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

    # Select random subset of 5 subjects
    subject_list = subject_list[:5]
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

    dice_array = np.array(DICE_scores)
    weighted_score = np.mean(dice_array) - np.std(dice_array)
    return weighted_score


# Run optimization
study = optuna.create_study(direction="maximize", study_name="Pituitary Segmentation")
study.optimize(objective, n_trials=100)

# Print best config
print("Best params:", study.best_params)
