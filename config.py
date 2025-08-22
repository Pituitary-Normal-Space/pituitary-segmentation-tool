# Description: Configuration file for the project.
# Allows you to manipulate the parameters of the preprocessing pipeline and NL space development without changing the main code.
# In the form of a singleton class to allow updating of parameters during runtime (used for tuning).

from typing import Literal

import numpy as np


class ModalityMaskParams:
    """
    Parameters specific to T1 or T2 modality-based masks.
    """

    def __init__(self, modality: Literal["t1", "t2"]):
        self.modality = modality

        if self.modality not in ["t1", "t2"]:
            raise ValueError(f"Invalid modality: {self.modality}")

        if self.modality == "t1":
            # Intensity-based params (can differ between T1 and T2)
            self.intensity_range = (400, 600)

            # Score weighting
            self.distance_weight = 0.5
            self.intensity_range_weight = 0.35
            self.connectivity_weight = 0.15
            self.min_score_threshold = 0.65

            # Connectivity
            self.high_quality_neighbors_to_consider_connected = 10

            # Region growing
            self.max_voxels = 8000
            self.region_growing_weight = 0
            self.score_based_weight = 1

            # Boosting
            self.num_neighbors_required_to_boost = 7
            self.min_score_to_boost_if_quality_neighbors = 0.7
            self.min_score_considered_high_score = 0.67

        else:
            # Intensity-based params (can differ between T1 and T2)
            self.intensity_range = (200, 300)

            # Score weighting
            self.distance_weight = 0.5
            self.intensity_range_weight = 0.35
            self.connectivity_weight = 0.15
            self.min_score_threshold = 0.65

            # Connectivity
            self.high_quality_neighbors_to_consider_connected = 10

            # Region growing
            self.region_growing_weight = 0
            self.score_based_weight = 1

            # Boosting
            self.num_neighbors_required_to_boost = 7
            self.min_score_to_boost_if_quality_neighbors = 0.7
            self.min_score_considered_high_score = 0.67


class Config:
    def __init__(self):
        # Parameters to optimize computer resources
        # Percent of CPUs to attempt to use for multiprocessing. Be weary of using too many CPUs as it can cause the computer to freeze d/t memory usage.
        self.percent_cpus_to_use = 0.2
        # This is the number of threads to use for FNIRT.
        self.number_threads_fnirt = 6  # I placed it at 8, I would leave it here. If you run into memory issues, you can try lowering it.

        # If you want to see images for every step of the pipeline, set this to True.
        # It will stop and wait for you to close the image to continue.
        self.show_images = False
        # Delete the temporary files created during the pipeline.
        self.delete_temp_files = True

        ####################################################
        # Pituitary Segmentation Parameters
        ####################################################
        # Shared params (do not differ between T1/T2)
        # The default centroid of the pituitary gland in MNI space. This is where we start our search for each subject.
        self.default_centroid = (
            0,
            2,
            -32,
        )  # Uses MNI space coordinates. Recommended based off this paper: https://core.ac.uk/download/pdf/288028975.pdf
        # This range of values is the X, Y, and Z coordinates of the pituitary gland in the MRI image.
        # These are values of MNI space coordinates.
        self.x_range = (-10, 12)  # Width of 24
        self.y_range = (-3, 10)  # Height of 12
        self.z_range = (-41, -27)  # Depth of 14

        self.max_voxel_drift = 2  # Number of voxels the centroid can drift in one direction before terminating the search

        # This is for region growing
        self.intensity_tolerance = 225
        self.max_voxels = 8000

        # Appendage removal
        self.do_appendage_removal = True
        self.infundibulum_range = 5
        self.appendage_removal_radius = 5

        # Modality-specific params
        self.t1 = ModalityMaskParams("t1")
        self.t2 = ModalityMaskParams("t2")

        # Voxel weights weighing T1 and T2 masks into a final score
        self.voxel_weights = self._init_voxel_weight_tensor()

        self.final_score_threshold = 0.32

        ####################################################
        # Preprocessing parameters
        ####################################################
        #
        ####################################################
        # Brain extraction parameters... NOT CURRENTLY IN USE
        #
        # Fractional intensity threshold (0-1) for BET (0.5 is default).
        self.fractional_intensity_t1 = 0.075
        self.fractional_intensity_t2 = 0.075
        # Gradient threshold for BET (higher values give larger brain outline estimates).
        self.gradient_t1 = 0.25
        self.gradient_t2 = 0.25
        # Robust brain extraction for BET (reduce sensitivity to noise and intensity bias).
        self.robust_brain_extraction = True
        #
        #
        ####################################################
        # Smoothing parameters
        #
        # Sigma value for Gaussian smoothing filter (controls blurriness of the image).
        # Small σ (e.g., 0.5 - 1.5) → Less smoothing, preserves fine details.
        # Moderate σ (e.g., 2 - 3) → Balanced smoothing, reduces noise while keeping structure.
        # Large σ (e.g., 5 - 10) → Heavy smoothing, blurs small details.
        self.smoothing_sigma = 0.75
        #
        #
        ####################################################
        # Non-linear registration parameters
        #
        # Specifies the order of the B-spline functions modelling the warp-fields. A spline-function is a piecewise continuous polynomial-function and the order of the spline determines the order of the polynomial and the support of the spline. In fnirt one can use splines of order 2 (quadratic) or 3 (the "well known" cubic B-spline).
        self.spline_order = 3  # 2 or 3, default is 3 but 2 is faster
        # Its value can be either float or double (default) and it specifies the precision that the hessian H is calculated and stored in. Changing this to float will decrease the amount of RAM needed to store H and will hence allow one to go to slightly higher warp-resolution. The default is double since that is what we have used for most of the testing and validation.
        self.hessian_precision = "double"  # float or double, default is dobule but float is faster and takes less ram

    def _init_voxel_weight_tensor(self) -> np.ndarray:
        """
        Create a 4D tensor with shape (X, Y, Z, 2),
        where the last axis stores [T1_weight, T2_weight].
        """
        x_dim = 45
        y_dim = 25
        z_dim = 29

        # Initialize with equal weights (0.5, 0.5)
        tensor = np.ones((x_dim, y_dim, z_dim, 2)) * 0.5
        return tensor

    def pad_voxel_weights(self, prob_mask_shape, mni_coords, img):
        """
        Pads/crops voxel_weights to align with ROI in voxel space,
        preserving its actual MNI spatial position.
        """
        # Convert MNI ROI coordinates into voxel indices
        affine = img.affine
        mni_to_voxel = np.linalg.inv(affine)
        voxel_min = np.dot(mni_to_voxel, np.array([*mni_coords[0], 1]))[:3]
        voxel_max = np.dot(mni_to_voxel, np.array([*mni_coords[1], 1]))[:3]
        voxel_min = np.round(voxel_min).astype(int)
        voxel_max = np.round(voxel_max).astype(int)
        voxel_min, voxel_max = np.minimum(voxel_min, voxel_max), np.maximum(
            voxel_min, voxel_max
        )

        # Calculate the shape needed for this ROI
        crop_shape = (
            voxel_max[0] - voxel_min[0] + 1,
            voxel_max[1] - voxel_min[1] + 1,
            voxel_max[2] - voxel_min[2] + 1,
            2,
        )

        # Check if we have ROI-specific weights from optimization
        if hasattr(self, "roi_voxel_weights") and self.roi_voxel_weights is not None:
            # Use the pre-generated ROI weights
            roi_weights = self.roi_voxel_weights

            # The roi_weights should match the crop_shape (minus the channel dimension)
            if roi_weights.shape[:3] == crop_shape[:3]:
                cropped_weights = roi_weights
            else:
                # If shapes don't match exactly, we might need to resize or crop
                # For now, fall back to uniform weights
                print(
                    f"Warning: ROI weights shape {roi_weights.shape[:3]} doesn't match crop shape {crop_shape[:3]}"
                )
                cropped_weights = np.full(crop_shape, 0.5, dtype=np.float32)
        else:
            # Fallback: create uniform weights using self attributes since this is a config method
            cropped_weights = np.full(crop_shape, 0.5, dtype=np.float32)

        print("Cropped weights shape:", cropped_weights.shape)

        # Allocate zero-filled tensor in full voxel space
        padded_weights = np.zeros((*prob_mask_shape, 2), dtype=cropped_weights.dtype)

        # Place cropped weights back in correct voxel coordinates
        x0, y0, z0 = voxel_min
        x1 = x0 + cropped_weights.shape[0]
        y1 = y0 + cropped_weights.shape[1]
        z1 = z0 + cropped_weights.shape[2]
        padded_weights[x0:x1, y0:y1, z0:z1, :] = cropped_weights

        # Store the aligned voxel weights
        self.voxel_weights = padded_weights

        print("Padded weights shape:", padded_weights.shape)
        print("Non-zero elements:", np.count_nonzero(padded_weights))

        return padded_weights

    def set_voxel_weight(
        self, x: int, y: int, z: int, t1_weight: float, t2_weight: float
    ):
        """
        Set per-voxel weighting for blending T1/T2 scores.
        """
        assert (
            abs((t1_weight + t2_weight) - 1.0) < 1e-6
        ), "T1 + T2 weights must sum to 1."
        self.voxel_weights[x, y, z] = [t1_weight, t2_weight]

    def get_voxel_weight(self, x: int, y: int, z: int) -> tuple[float, float]:
        """
        Get the (T1, T2) weights for a given voxel.
        """
        return self.voxel_weights[x, y, z]


# Create a singleton instance
config = Config()
