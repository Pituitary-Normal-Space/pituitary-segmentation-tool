# Built in imports
import os

# Internal imports
from const import PATH_TO_GROUND_TRUTH_MASKS, KEY_MAP_NAME, PATH_TO_UNPROCESSED_DATA
from config import config

# External imports
import optuna
import numpy as np
import pandas as pd


def gaussian_field(shape, centers, sigmas, alphas):
    """Generate a smooth field from Gaussian bumps."""
    X, Y, Z = np.indices(shape)
    field = np.zeros(shape, dtype=np.float32)

    for (cx, cy, cz), sigma, alpha in zip(centers, sigmas, alphas):
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
        field += alpha * np.exp(-dist2 / (2 * sigma**2))

    return field


def run_segmentation(subject) -> str:
    """
    Run the segmentation pipeline for a given subject.

    :param subject: Subject object containing MRI data and configuration.
    :return: string path to the predicted mask.
    """
    subject.setup_pituitary_analysis()
    subject.segment_pituitary_gland(weighted_img_to_use="both")
    return subject.binary_pituitary_mask


def objective(trial) -> float:
    """
    Objective function for Optuna optimization.
    :param trial: Optuna trial object.

    :return: Weighted average of DICE scores.
    """
    print("Trial:", trial.number)

    # Tune voxel weights by modality T1 vs T2
    shape = config.voxel_weights.shape[:3]
    # Tune number of gaussian control points
    n_gaussians = trial.suggest_int("n_gaussians", 1, 5)  # Number of control points

    centers = []
    sigmas = []
    alphas = []
    for i in range(n_gaussians):
        cx = trial.suggest_int(f"cx_{i}", 0, shape[0] - 1)
        cy = trial.suggest_int(f"cy_{i}", 0, shape[1] - 1)
        cz = trial.suggest_int(f"cz_{i}", 0, shape[2] - 1)
        sigma = trial.suggest_float(f"sigma_{i}", 5, 20)
        alpha = trial.suggest_float(f"alpha_{i}", 0.0, 1.0)
        centers.append((cx, cy, cz))
        sigmas.append(sigma)
        alphas.append(alpha)

    # Build T1 field
    t1_field = gaussian_field(shape, centers, sigmas, alphas)
    # Normalize
    t1_field = (t1_field - t1_field.min()) / (t1_field.max() - t1_field.min() + 1e-8)
    t2_field = 1 - t1_field

    voxel_weights = np.stack([t1_field, t2_field], axis=-1)

    # Save to config
    config.voxel_weights = voxel_weights

    # Save weights for this trial
    np.save(f"voxel_weights/voxel_weights_trial_{trial.number}.npy", voxel_weights)

    # Now tune intensity range
    minimum = trial.suggest_int("intensity_range_t1__min", 100, 500)
    maximum = trial.suggest_int("intensity_range_t1__max", 500, 1000)
    config.t1.intensity_range = (minimum, maximum)

    minimum = trial.suggest_int("intensity_range_t2__min", 50, 225)
    maximum = trial.suggest_int("intensity_range_t2__max", 225, 500)
    config.t2.intensity_range = (minimum, maximum)

    # Tune voxel drift
    # config.max_voxel_drift = trial.suggest_int("max_voxel_drift", 0, 3)
    config.max_voxel_drift = 1

    # Tune score based weights - using only two parameters and calculating the third
    config.t1.distance_weight = trial.suggest_float("distance_weight_t1", 0, 1)
    config.t1.intensity_range_weight = trial.suggest_float(
        "intensity_range_weight_t1", 0, 1 - config.t1.distance_weight
    )
    # Connectivity weight is automatically set to complement to 1
    config.t1.connectivity_weight = 1 - (
        config.t1.distance_weight + config.t1.intensity_range_weight
    )

    config.t2.distance_weight = trial.suggest_float("distance_weight_t2", 0, 1)
    config.t2.intensity_range_weight = trial.suggest_float(
        "intensity_range_weight_t2", 0, 1 - config.t2.distance_weight
    )
    config.t2.connectivity_weight = 1 - (
        config.t2.distance_weight + config.t2.intensity_range_weight
    )

    # Tune "high quality" neighbors
    config.t1.high_quality_neighbors_to_consider_connected = trial.suggest_int(
        "high_quality_neighbors_to_consider_connected_t1", 2, 10
    )
    config.t2.high_quality_neighbors_to_consider_connected = trial.suggest_int(
        "high_quality_neighbors_to_consider_connected_t2", 2, 10
    )

    # Tune min score threshold
    config.t1.min_score_threshold = trial.suggest_float(
        "min_score_threshold_t1", 0.45, 0.95
    )
    config.t2.min_score_threshold = trial.suggest_float(
        "min_score_threshold_t2", 0.45, 0.95
    )

    # # Tune intensity tolerance
    # config.intensity_tolerance = trial.suggest_int("intensity_tolerance", 10, 500)
    # # Tune max voxels
    # config.max_voxels = trial.suggest_int("max_voxels", 10000, 20000)
    # Tune region growing weight
    # config.score_based_weight = trial.suggest_float("score_based_weight", 0, 1)
    config.score_based_weight = 1
    config.region_growing_weight = (
        1 - config.score_based_weight  # Automatically complements to 1
    )

    # config.t1.num_neighbors_required_to_boost = trial.suggest_int(
    #     "num_neighbors_required_to_boost_t1", 4, 26
    # )
    # config.t1.min_score_to_boost_if_quality_neighbors = trial.suggest_float(
    #     "min_score_to_boost_if_quality_neighbors_t1", 0, 1
    # )
    # config.t1.min_score_considered_high_score = trial.suggest_float(
    #     "min_score_considered_high_score_t1", 0.5, 1
    # )

    # config.t2.num_neighbors_required_to_boost = trial.suggest_int(
    #     "num_neighbors_required_to_boost_t2", 4, 26
    # )
    # config.t2.min_score_to_boost_if_quality_neighbors = trial.suggest_float(
    #     "min_score_to_boost_if_quality_neighbors_t2", 0, 1
    # )
    # config.t2.min_score_considered_high_score = trial.suggest_float(
    #     "min_score_considered_high_score_t2", 0.5, 1
    # )

    # Tune appendage removal
    config.do_appendage_removal = trial.suggest_categorical(
        "do_appendage_removal", [True, False]
    )

    # Tune infundibulum range
    config.infundibulum_range = trial.suggest_int("infundibulum_range", 0, 10)

    # Tune appendage removal radius
    config.appendage_removal_radius = trial.suggest_int(
        "appendage_removal_radius", 4, 6
    )

    # Tune final score threshold
    config.final_score_threshold = trial.suggest_float(
        "final_score_threshold", 0.1, 0.9
    )

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

    # Select only men
    subject_list = [subject for subject in subject_list if subject.sex == "M"]
    print("Selected subjects:", [subject.subject_id for subject in subject_list])

    # Sample 70 percent and leave 30% for testing
    train_subjects = subject_list[: int(len(subject_list) * 0.7)]
    test_subjects = subject_list[int(len(subject_list) * 0.7) :]

    print("Training subjects:", [subject.subject_id for subject in train_subjects])
    print("Testing subjects:", [subject.subject_id for subject in test_subjects])

    # Run segmentation for each subject and collect predicted masks
    predicted_masks = {
        subject.subject_id: run_segmentation(subject) for subject in train_subjects
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

    # Write average DICE score to a file
    with open("average_DICE_score.txt", "a") as f:  # Changed 'w' to 'a' for append mode
        f.write(f"{trial.number} {str(average_DICE_score)}\n")  # Added newline characte

    dice_array = np.array(DICE_scores)
    weighted_score = np.mean(dice_array) - np.std(dice_array)
    return weighted_score


# Run optimization
study = optuna.create_study(
    direction="maximize",
    study_name="Pituitary Segmentation Joint: Score-Based Only",
    storage="sqlite:///pituitary_segmentation_score_only_joint.db",
    load_if_exists=True,
)
study.optimize(objective, n_trials=300)

# Print best config
print("Best params:", study.best_params)
