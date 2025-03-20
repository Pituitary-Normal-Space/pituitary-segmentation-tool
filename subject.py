"""
File that contains the Subject class. This is a class that represents a subject, their details, and stores their MRI data.
"""

# Standard libs
import os
import subprocess
from typing import Literal, Tuple, Dict, Optional, Any

# Non-standard libs
import numpy as np
import nibabel as nib
from scipy.ndimage import label, convolve
from skimage.morphology import ball, binary_closing

# Local libs
from config import config

from const import (
    # T1_BRAIN_FILE,
    # T2_BRAIN_FILE,
    T1_MC_FILE,
    T2_MC_FILE,
    T1_SMOOTH_FILE,
    T2_SMOOTH_FILE,
    T1_SMOOTH_REGISTERED_FILE,
    T1_TO_T2_MAT_FILE,
    T1_MC_REGISTERED_FILE,
    T1_AFFINE_TO_MNI_FILE,
    T1_SMOOTH_NONLIN_TO_MNI_FILE,
    T1_TO_MNI_WARP_FILE,
    T1_MNI_FILE,
    T2_MNI_FILE,
    OVERLAY_FILE,
    PITUITARY_MASK_FILE,
    T1_MNI_MAT_FILE,
    PATH_TO_PROCESSED_DATA,
    MNI_TEMPLATE,
    PITUITARY_CENTROID_FILE,
    SUFFIX_SUBJECT_ID_PROCESSED,
    T1_TO_T2_WARP_FILE,
    PROB_MASK_PREFIX,
    REGION_GROWING_MASK_PREFIX,
    PROB_PITUITARY_MASK,
)
from mri import show_mri_slices


class Subject:
    """
    Subject class that represents a subject, their details, and stores their MRI data.

    :param subject_id: The ID of the subject.
    :param age: The age range of the subject.
    :param sex: The sex of the subject.
    :param t1_path: The path to the T1 MRI data.
    :param t2_path: The path to the T2 MRI data.

    :return: None
    """

    def __init__(
        self,
        subject_id: str,
        age: Literal[
            "31-35", "26-30", "22-25"
        ],  # As pre-determined by the Human Connectome Project
        sex: Literal["M", "F"],
        t1_path: str,
        t2_path: str,
    ) -> None:
        # Id and demographic information
        self.subject_id: str = str(subject_id)
        self.age: Literal["31-35", "26-30", "22-25"] = age
        self.sex: Literal["M", "F"] = sex

        # Flags to track processing status
        self.preproc_complete: bool = False
        self.moved_to_mni_norm_space: bool = False
        self.mask_complete: bool = False

        # Unprocessed data
        # Paths to MRI data
        self.unprocessed_t1: Optional[str] = t1_path
        self.unprocessed_t2: Optional[str] = t2_path

        # Processed files
        # The directory that has this subject's processed data
        self.output_dir: Optional[str] = None
        # Paths to final processed MRI
        self.t1_in_mni_space: Optional[str] = None
        self.t2_in_mni_space: Optional[str] = None
        # Paths to intermediate processed MRI
        self.motion_corrected_t1w: Optional[str] = None
        self.motion_corrected_t2w: Optional[str] = None
        self.smoothed_and_mc_t1w: Optional[str] = None
        self.smoothed_and_mc_t2w: Optional[str] = None
        self.t1w_smooth_reg_t2: Optional[str] = None  # T1 smoothed and registered to T2
        self.t1w_preproc_reg_t2: Optional[str] = (
            None  # T1 not smoothed and registered to T2
        )
        self.overlayed_t1_and_t2: Optional[str] = None

        # Paths to transformation matrices and warp fields
        self.affine_matrix: Optional[str] = None
        self.warp_field: Optional[str] = None

        # Paths to pituitary mask and dict of statistics
        self.score_based_mask_scores: Optional[str] = None
        self.binary_pituitary_mask: Optional[str] = None
        self.prob_pituitary_mask: Optional[str] = None
        self.final_mask_stats: Optional[Dict[str, Any]] = None

        # Check that config parameters are valid
        # If number_threads_fnirt is not a string representation of an int, raise an error
        if type(config.number_threads_fnirt) is not int:
            raise ValueError("Number of threads for FNIRT must be a int.")

        if config.number_threads_fnirt < 1:
            raise ValueError("Number of threads for FNIRT must be greater than 0.")

        if (
            type(config.x_range) is not tuple
            or len(config.x_range) != 2
            and type(config.x_range[0]) is not int
            and type(config.x_range[1]) is not int
        ):
            raise ValueError("X range must be a tuple of two integers")

        if (
            type(config.y_range) is not tuple
            or len(config.y_range) != 2
            and type(config.y_range[0]) is not int
            and type(config.y_range[1]) is not int
        ):
            raise ValueError("Y range must be a tuple of two integers")

        if (
            type(config.z_range) is not tuple
            or len(config.z_range) != 2
            and type(config.z_range[0]) is not int
            and type(config.z_range[1]) is not int
        ):
            raise ValueError("Z range must be a tuple of two integers")

        if (
            type(config.intensity_range) is not tuple
            or len(config.intensity_range) != 2
            and type(config.intensity_range[0]) is not int
            and type(config.intensity_range[1]) is not int
        ):
            raise ValueError("Intensity range must be a tuple of two integers")

        if type(config.distance_weight) is not float:
            raise ValueError("Distance weight must be a float")

        if type(config.intensity_range_weight) is not float:
            raise ValueError("Intensity range weight must be a float")

        if type(config.connectivity_weight) is not float:
            raise ValueError("Connectivity weight must be a float")

        if type(config.min_score_threshold) is not float:
            raise ValueError("Minimum score threshold must be a float between 0 and 1")

        if type(config.intensity_tolerance) is not int:
            raise ValueError("Intensity tolerance must be an integer")

        if config.intensity_tolerance < 0:
            raise ValueError("Intensity tolerance must be greater than 0")

        if type(config.max_voxels) is not int:
            raise ValueError("Max voxels must be an integer")

        if config.max_voxels < 0:
            raise ValueError("Max voxels must be greater than 0")

        if config.min_score_threshold < 0 or config.min_score_threshold > 1:
            raise ValueError("Minimum score threshold must be between 0 and 1")

        if (
            round(
                config.distance_weight
                + config.intensity_range_weight
                + config.connectivity_weight,
                10,
            )
            != 1
        ):
            raise ValueError(
                f"Weights for distance, intensity, connectivity, and naive mask must add up to 1 they are {config.distance_weight}, {config.intensity_range_weight}, {config.connectivity_weight} adding to {config.distance_weight + config.intensity_range_weight + config.connectivity_weight}"
            )

        if type(config.fractional_intensity_t1) is not float:
            raise ValueError("Fractional intensity threshold for T1 must be a float")

        if type(config.fractional_intensity_t2) is not float:
            raise ValueError("Fractional intensity threshold for T2 must be a float")

        if config.fractional_intensity_t1 < 0 or config.fractional_intensity_t1 > 1:
            raise ValueError(
                "Fractional intensity threshold for T1 must be between 0 and 1"
            )

        if config.fractional_intensity_t2 < 0 or config.fractional_intensity_t2 > 1:
            raise ValueError(
                "Fractional intensity threshold for T2 must be between 0 and 1"
            )

        if type(config.gradient_t1) is not float:
            raise ValueError("Gradient threshold for T1 must be a float")

        if type(config.gradient_t2) is not float:
            raise ValueError("Gradient threshold for T2 must be a float")

        if config.gradient_t1 < 0:
            raise ValueError("Gradient threshold for T1 must be greater than 0")

        if config.gradient_t2 < 0:
            raise ValueError("Gradient threshold for T2 must be greater than 0")

        if type(config.robust_brain_extraction) is not bool:
            raise ValueError("Robust brain extraction must be a boolean")

        if (
            type(config.smoothing_sigma) is not float
            and type(config.smoothing_sigma) is not int
        ):
            raise ValueError("Smoothing sigma must be an integer")

        if config.smoothing_sigma < 0:
            raise ValueError("Smoothing sigma must be greater than 0")

        if config.spline_order not in [2, 3]:
            raise ValueError("Spline order must be 2 or 3")

        if config.hessian_precision not in ["double", "float"]:
            raise ValueError("Hessian precision must be double or float")

        if type(config.delete_temp_files) is not bool:
            raise ValueError("Delete temp files must be a boolean")

        if (
            type(config.default_centroid) is not tuple
            or len(config.default_centroid) != 3
        ):
            raise ValueError("Default centroid must be a tuple of three integers")

        if (
            type(config.default_centroid[0]) is not int
            or type(config.default_centroid[1]) is not int
            or type(config.default_centroid[2]) is not int
        ):
            raise ValueError("Default centroid must be a tuple of three integers")

        if type(config.max_voxel_drift) is not int:
            raise ValueError("Max voxel drift must be an integer")

        if config.max_voxel_drift < 0:
            raise ValueError("Max voxel drift must be greater than 0")

        if type(config.high_quality_neighbors_to_consider_connected) is not int:
            raise ValueError(
                "High quality neighbors to consider connected must be an integer"
            )

        if (
            config.high_quality_neighbors_to_consider_connected < 0
            or config.high_quality_neighbors_to_consider_connected > 26
        ):
            raise ValueError(
                "High quality neighbors to consider connected must be greater than 0 and less than 26"
            )

        if (
            type(config.region_growing_weight) is not float
            and type(config.region_growing_weight) is not int
        ):
            raise ValueError("Region growing weight must be a float")

        if (
            type(config.score_based_weight) is not float
            and type(config.score_based_weight) is not int
        ):
            raise ValueError("Score based weight must be a float")

        if config.region_growing_weight < 0 or config.region_growing_weight > 1:
            raise ValueError("Region growing weight must be between 0 and 1")

        if config.score_based_weight < 0 or config.score_based_weight > 1:
            raise ValueError("Score based weight must be between 0 and 1")

        if round(config.score_based_weight + config.region_growing_weight != 1, 10):
            raise ValueError(
                "Region growing weight and score based weight must add up to 1"
            )

        if type(config.num_neighbors_required_to_boost) is not int:
            raise ValueError("Number of neighbors required to boost must be an integer")

        if (
            config.num_neighbors_required_to_boost < 0
            or config.num_neighbors_required_to_boost > 26
        ):
            raise ValueError(
                "Number of neighbors required to boost must be between 0 and 26"
            )

        if type(config.min_score_considered_high_score) is not float:
            raise ValueError("Minimum score considered high score must be a float")

        if (
            config.min_score_considered_high_score < 0
            or config.min_score_considered_high_score > 1
        ):
            raise ValueError(
                "Minimum score considered high score must be between 0 and 1"
            )

        if type(config.min_score_to_boost_if_quality_neighbors) is not float:
            raise ValueError(
                "Minimum score to boost if quality neighbors must be a float"
            )

        if (
            config.min_score_to_boost_if_quality_neighbors < 0
            or config.min_score_to_boost_if_quality_neighbors > 1
        ):
            raise ValueError(
                "Minimum score to boost if quality neighbors must be between 0 and 1"
            )

        if type(config.infundibulum_range) is not int:
            raise ValueError("Infundibulum range must be an integer")

        if config.infundibulum_range < 0:
            raise ValueError("Infundibulum range must be greater than 0")

        if (
            type(config.appendage_removal_radius) is not float
            and type(config.appendage_removal_radius) is not int
        ):
            raise ValueError("Appendage removal radius must be a float or int")

        if config.appendage_removal_radius < 1.0:
            raise ValueError("Appendage removal radius must be greater than 1")

        if type(config.final_score_threshold) is not float:
            raise ValueError("Final score threshold must be a float")

        if config.final_score_threshold < 0 or config.final_score_threshold > 1:
            raise ValueError("Final score threshold must be between 0 and 1")

        if type(config.do_appendage_removal) is not bool:
            raise ValueError("Do appendage removal must be a boolean")

    def __str__(self) -> str:
        """
        String representation of the subject object.

        :return: A string containing the subject's details and MRI processing status.
        """
        # Every subject has these details
        base_info = (
            f"Subject ID: {self.subject_id}\n"
            f"Age: {self.age}\n"
            f"Sex: {self.sex}\n"
            f"Processing Complete: {self.preproc_complete}\n"
            f"Moved to MNI Space: {self.moved_to_mni_norm_space}\n"
            f"Pituitary Mask Complete: {self.mask_complete}\n\n"
            "MRI Paths:\n"
            f"UnProcessed T1 Path: {self.unprocessed_t1}\n"
            f"UnProcessed T2 Path: {self.unprocessed_t2}"
        )

        # If preprocessing is complete, add processed MRI paths
        if not self.preproc_complete and not self.moved_to_mni_norm_space:
            return base_info

        processed_info = (
            f"\nProcessed T1 MNI Path: {self.t1_in_mni_space}\n"
            f"Processed T2 MNI Path: {self.t2_in_mni_space}\n"
        )

        # If pituitary mask is created, add mask details
        if self.mask_complete and self.prob_pituitary_mask and self.final_mask_stats:
            mask_info = (
                "Pituitary Segmentation Statistics:\n"
                f"\nPituitary Mask Path: {self.prob_pituitary_mask}\n"
                f"Pituitary Statistics:\n"
                f"  Volume (voxels): {self.final_mask_stats['volume_voxels']}\n"
                f"  Volume (mm³): {self.final_mask_stats['volume_mm3']:.2f}\n"
                f"  Center (MNI): [{', '.join(f'{x:.2f}' for x in self.final_mask_stats['center_mni'])}]\n"
                f"  Mean Intensity: {self.final_mask_stats['mean_intensity']:.2f}\n"
                f"  Min Intensity: {self.final_mask_stats['min_intensity']:.2f}\n"
                f"  Max Intensity: {self.final_mask_stats['max_intensity']:.2f}\n"
                f"  Std Intensity: {self.final_mask_stats['std_intensity']:.2f}"
            )
            return base_info + processed_info + mask_info

        return base_info + processed_info

    def setup_pituitary_analysis(self) -> None:
        """
        This function assumes that the MRI data has been preprocessed and registered to MNI space.

        It sets up the necessary files and directories for pituitary analysis.

        :return: None
        """
        # The output dir should exist we just need to save the path.
        self.__create_output_directory()

        # Set flags
        self.preproc_complete = True
        self.moved_to_mni_norm_space = True

        # Grab the paths to the processed MRIs
        self.t1_in_mni_space = os.path.join(self.output_dir, T1_MNI_FILE)
        self.t2_in_mni_space = os.path.join(self.output_dir, T2_MNI_FILE)

    def preprocess_MRIs(self) -> None:
        """
        Preprocesses T1w_MPR and T2w_SPC MRIs using:
        - FAST (FSL’s automated segmentation tool) for tissue segmentation (gray matter, white matter, CSF).
        - FEAT (FSL’s motion correction & smoothing) for motion correction & Gaussian smoothing.
        - FLIRT (FSL’s linear registration) for affine alignment.
        - FNIRT (FSL’s nonlinear registration) for fine-grained normalization.
        The final processed images will be stored in a structured output directory.

        What it does step by step:
        1. Motion correction and smoothing of T1 and T2 images.
        2. Registration of T1 to T2 using FLIRT and smoothed images. Moving the MC T1 to T2 space.

        :return: None. Saves the preprocessed images in the output directory.
        """
        if self.preproc_complete:
            print("Preprocessing already completed. Skipping...")
            return

        self.__create_output_directory()
        # self.__perform_brain_extraction()
        self.__motion_correction_and_smoothing()
        self.__registration_and_normalization()

        if config.delete_temp_files:
            self.__delete_temp_files("preprocessed")

        print(
            "Preprocessing completed successfully. This means that the T1W image is in T2W image space and both are motion corrected. Outputs stored in:",
            self.output_dir,
        )

        self.preproc_complete = True

    def overlay_MRIs(self) -> None:
        """
        Function to overlay the T1 and T2 MRI data of the subject.
        This step is performed **after** preprocessing is complete and is not necessary for pituitary analysis.
        The overlay allows for a visual comparison between the two modalities.

        :return: None. Saves the overlayed image in the output directory.
        """
        if not self.preproc_complete:
            raise ValueError("Preprocessing not complete. Run preprocess_MRIs first.")

        # Construct file paths for the preprocessed images
        self.overlayed_t1_and_t2 = os.path.join(self.output_dir, OVERLAY_FILE)

        # Ensure the preprocessed images exist before overlaying
        if not os.path.exists(self.t1w_preproc_reg_t2) or not os.path.exists(
            self.motion_corrected_t2w
        ):
            raise FileNotFoundError(
                "Preprocessed images not found. Run preprocess_MRIs first."
            )

        # Overlay T1w and T2w images using FSL’s fslmaths
        subprocess.run(
            [
                "fslmaths",
                self.t1w_preproc_reg_t2,
                "-add",
                self.motion_corrected_t2w,
                self.overlayed_t1_and_t2,
            ],
            check=True,
        )

        # Show a slice of the overlayed image
        show_mri_slices([self.overlayed_t1_and_t2], titles=["T1w T2w Overlay MRI"])

        print(f"Overlaying MRIs completed. Output saved at {self.overlayed_t1_and_t2}.")

    def coregister_to_mni_space(
        self,
        mni_template_path: str = MNI_TEMPLATE,
    ) -> None:
        """
        Coregisters both the T1 and T2 images to the MNI template using affine and nonlinear warping.

        The transformation is computed on the smoothed T1 image, but the transformation is later
        applied to the non-smoothed, motion-corrected T1 and T2 images to avoid excessive blurring.
        """
        if not self.preproc_complete:
            raise ValueError("Preprocessing not complete. Run preprocess_MRIs first.")

        if self.moved_to_mni_norm_space:
            print("Already moved to MNI space. Skipping registration.")
            return

        if type(mni_template_path) is not str:
            raise ValueError("MNI Template path must be a string.")

        if not os.path.exists(mni_template_path):
            raise FileNotFoundError(
                "MNI Template not found. Please provide a valid path to the MNI template."
            )

        # Paths to save transformations
        self.affine_matrix = os.path.join(self.output_dir, T1_MNI_MAT_FILE)
        self.warp_field = os.path.join(self.output_dir, T1_TO_MNI_WARP_FILE)

        # Output paths for final non-blurry normalized images
        self.t1_in_mni_space = os.path.join(self.output_dir, T1_MNI_FILE)
        self.t2_in_mni_space = os.path.join(self.output_dir, T2_MNI_FILE)

        # If not previously coregistered and warp field exists, skip registration
        if not os.path.exists(self.warp_field):
            print(
                "Transforming to MNI space using smoothed T1 image... Starting with affine linear registration."
            )

            affine_to_mni = os.path.join(self.output_dir, T1_AFFINE_TO_MNI_FILE)

            # Compute Affine Transformation using FLIRT
            subprocess.run(
                [
                    "flirt",
                    "-in",
                    self.t1w_smooth_reg_t2,  # Smoothed T1 used for registration
                    "-ref",
                    mni_template_path,
                    "-out",
                    affine_to_mni,
                    "-omat",
                    self.affine_matrix,
                ],
                check=True,
            )

            print(
                "Affine registration to MNI completed. Starting non-linear registration. This may take a while..."
            )

            smoothed_normalized = os.path.join(
                self.output_dir, T1_SMOOTH_NONLIN_TO_MNI_FILE
            )

            env = os.environ.copy()
            env["OMP_NUM_THREADS"] = str(config.number_threads_fnirt)
            # Compute Nonlinear Warp using FNIRT
            subprocess.run(
                [
                    "fnirt",
                    f"--in={self.t1w_smooth_reg_t2}",
                    f"--ref={mni_template_path}",
                    f"--aff={self.affine_matrix}",
                    f"--cout={self.warp_field}",
                    f"--iout={smoothed_normalized}",
                    f"--splineorder={str(config.spline_order)}",
                    f"--numprec={config.hessian_precision}",
                    "--verbose",
                ],
                check=True,
            )

            print(
                f"Nonlinear registration (FNIRT) to MNI completed. Warp file is saved at {self.warp_field}"
            )

            # Clean up intermediate files
            os.remove(affine_to_mni)
            os.remove(smoothed_normalized)

        # Step 3: Apply the Transformation to Non-blurry Motion-Corrected Images

        print(
            "Applying warp transformations to motion-corrected images for T1 and T2 images..."
        )

        # Apply affine + nonlinear warp to motion-corrected T1
        subprocess.run(
            [
                "applywarp",
                "-i",
                self.t1w_preproc_reg_t2,  # Non-smoothed motion-corrected T1
                "-r",
                mni_template_path,
                "-w",
                self.warp_field,
                "-o",
                self.t1_in_mni_space,
            ],
            check=True,
        )

        print(
            f"T1 Motion Corrected Image Registered to MNI. Located here: {self.t1_in_mni_space}"
        )

        # Apply affine + nonlinear warp to motion-corrected T2
        subprocess.run(
            [
                "applywarp",
                "-i",
                self.motion_corrected_t2w,  # Non-smoothed motion-corrected T2
                "-r",
                mni_template_path,
                "-w",
                self.warp_field,
                "-o",
                self.t2_in_mni_space,
            ],
            check=True,
        )

        print(
            f"T2 Motion Corrected Image Registered to MNI. Located here: {self.t2_in_mni_space}"
        )

        # Show results
        show_mri_slices(
            [self.t1_in_mni_space, self.t2_in_mni_space, mni_template_path],
            titles=[
                "T1w Motion Corrected Normalized to MNI",
                "T2w Motion Corrected Normalized to MNI",
                "MNI Template",
            ],
        )

        self.moved_to_mni_norm_space = True

        if config.delete_temp_files:
            self.__delete_temp_files("in_mni")

    def segment_pituitary_gland(
        self,
        mni_coords: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
            (config.x_range[0], config.y_range[0], config.z_range[0]),
            (config.x_range[1], config.y_range[1], config.z_range[1]),
        ),
        appendage_removal=config.do_appendage_removal,
    ) -> None:
        """
        Creates a probabilistic pituitary mask using both methods, refines it, and calculates statistics.

        :param mni_coords: A tuple containing two MNI coordinate bounds to define the search region.
        :param appendage_removal: A boolean to remove appendages from the mask.

        :return: None. Saves the final pituitary mask in the output directory. Statistics are stored in final_mask_stats.
        """
        if not self.moved_to_mni_norm_space and not self.preproc_complete:
            raise ValueError(
                "Images not in MNI space. Run coregister_to_mni_space first."
            )

        if self.mask_complete:
            print("Pituitary mask already created. Skipping...")
            return self.final_mask_stats

        if type(mni_coords) is not tuple or len(mni_coords) != 2:
            raise ValueError("MNI coordinates must be a tuple of two tuples.")

        if type(mni_coords[0]) is not tuple or type(mni_coords[1]) is not tuple:
            raise ValueError("MNI coordinates must be a tuple of two tuples.")

        if len(mni_coords[0]) != 3 or len(mni_coords[1]) != 3:
            raise ValueError("MNI coordinates must be a tuple of three integers.")

        if (
            type(mni_coords[0][0]) is not int
            or type(mni_coords[0][1]) is not int
            or type(mni_coords[0][2]) is not int
            or type(mni_coords[1][0]) is not int
            or type(mni_coords[1][1]) is not int
            or type(mni_coords[1][2]) is not int
        ):
            raise ValueError("MNI coordinates must be a tuple of three integers.")

        # List to store previous centroids to check for convergence
        previous_centroids = []
        centroid = config.default_centroid
        # Get current w, h, d around centroid using mni coords
        width = abs(mni_coords[1][0] - mni_coords[0][0])
        height = abs(mni_coords[1][1] - mni_coords[0][1])
        depth = abs(mni_coords[1][2] - mni_coords[0][2])
        while centroid not in previous_centroids:
            if len(previous_centroids) > 0:
                print(
                    f"Centroid has shifted for the {len(previous_centroids)}th time. Searching for pituitary gland around centroid: {centroid}"
                )
            previous_centroids.append(centroid)
            # If greater than max_voxel_drift mm3 shift in a direction from original centroid, break
            if self.__check_for_drift(
                config.default_centroid, centroid, config.max_voxel_drift
            ):
                print("Centroid has drifted too far from origin. Exiting...")
                break

            # Update MNI coordinates to search around the centroid, keep same w, h, d
            # This should complete a frame shift around the centroid
            mni_coords = self.__shift_pituitary_roi(centroid, width, height, depth)

            # Create your probabilistic mask first using your scoring method
            prob_mask = self.__create_probabilistic_pituitary_mask(
                mni_coords=mni_coords, centroid=centroid
            )

            # Save the probabilistic mask with the scores
            prob_mask_img = nib.Nifti1Image(
                prob_mask, nib.load(self.t1_in_mni_space).affine
            )
            self.prob_pituitary_mask = os.path.join(
                self.output_dir, PROB_MASK_PREFIX + PROB_PITUITARY_MASK
            )
            nib.save(prob_mask_img, self.prob_pituitary_mask)
            # Show the probabilistic mask overlayed on the T1 image
            show_mri_slices(
                [self.t1_in_mni_space, self.prob_pituitary_mask],
                slice_index=self.slice_indices,
                titles=[
                    "T1 MRI with Probabilistic Pituitary Mask Overlay Pre-Appendage Removal"
                ],
                overlay=True,
                colormaps=["gray", "viridis"],  # T1 image  # Probability map
            )
            t1_img = nib.load(self.t1_in_mni_space)
            if appendage_removal:
                # Remove appendages before converting to binary
                print("Refining mask by removing appendages...")
                refined_mask = self.__remove_appendages(prob_mask, t1_img.affine)

                # Save the refined mask with the compbined probabilities after appendage removal
                refined_mask_img = nib.Nifti1Image(refined_mask, t1_img.affine)
                self.prob_pituitary_mask = os.path.join(
                    self.output_dir, PROB_MASK_PREFIX + PROB_PITUITARY_MASK
                )
                nib.save(refined_mask_img, self.prob_pituitary_mask)

                # Show the refined mask overlayed on the T1 image
                show_mri_slices(
                    [self.t1_in_mni_space, self.prob_pituitary_mask],
                    slice_index=self.slice_indices,
                    titles=[
                        "T1 MRI with Refined Pituitary Mask Overlay Post-Appendage Removal"
                    ],
                    overlay=True,
                    colormaps=["gray", "viridis"],  # T1 image  # Probability map
                )

            else:
                refined_mask = prob_mask

            # Convert refined mask to binary for statistics using 0.5 threshold
            binary_mask_img = nib.Nifti1Image(
                (refined_mask > config.final_score_threshold).astype(np.uint8),
                t1_img.affine,
            )
            self.binary_pituitary_mask = os.path.join(
                self.output_dir, PITUITARY_MASK_FILE
            )
            nib.save(binary_mask_img, self.binary_pituitary_mask)

            # Get the centroid of the refined mask
            centroid = tuple(
                round(coor) for coor in self.__get_pituitary_statistics()["center_mni"]
            )

        # Show the final refined mask overlayed on the T1 image
        show_mri_slices(
            [self.t1_in_mni_space, self.binary_pituitary_mask],
            slice_index=self.slice_indices,
            titles=["Final T1 MRI with Refined Pituitary Mask Overlay"],
            overlay=True,
            colormaps=["gray", "viridis"],  # T1 image  # Probability map
        )

        self.mask_complete = True

    def __check_for_drift(
        self, original_centroid: tuple, centroid: tuple, drifted_value: int
    ) -> bool:
        """
        Checks if the centroid has drifted by a certain value from the original centroid.

        :param original_centroid: The original centroid to compare against.
        :param centroid: The centroid to check.
        :param drifted_value: The value to check for drift against.

        :return: A boolean indicating if the centroid has drifted.
        """
        if len(original_centroid) != len(centroid) and len(centroid) != 3:
            raise ValueError("Centroids must have the same number of dimensions--3.")

        if type(drifted_value) is not int:
            raise ValueError("Drifted value must be an integer.")

        if drifted_value < 0:
            raise ValueError("Drifted value must be greater than 0.")

        return any(
            abs(original - current) > drifted_value
            for original, current in zip(original_centroid, centroid)
        )

    def __shift_pituitary_roi(
        self, centroid: Tuple[int, int, int], width: int, height: int, depth: int
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Shift the MNI coordinates to search around the centroid for the pituitary gland.

        :param centroid: The centroid to search around.
        :param width: The width of the search region.
        :param height: The height of the search region.
        :param depth: The depth of the search region.

        :return: A tuple containing the shifted MNI coordinates.
        """
        return (
            (
                centroid[0] - width // 2,
                centroid[1] - height // 2,
                centroid[2] - depth // 2,
            ),
            (
                centroid[0] + width // 2,
                centroid[1] + height // 2,
                centroid[2] + depth // 2,
            ),
        )

    def __create_naive_pituitary_mask(
        self,
        mni_coords: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
            (config.x_range[0], config.y_range[0], config.z_range[0]),
            (config.x_range[1], config.y_range[1], config.z_range[1]),
        ),  # These coordinates were determined by me
    ) -> None:
        """
        Grabs the ROI of MNI coordinates and creates a naive pituitary mask based on intensity within a range.

        :param mni_coords: A tuple containing two MNI coordinate bounds to define the search region.

        :return: None. Saves the naive pituitary mask in the output directory.
        """
        print("Creating naive pituitary mask...")
        if not self.t1_in_mni_space:
            raise FileNotFoundError(
                "T1 MRI image in MNI space not found. Run coregister_to_mni_space first."
            )

        # Define the output mask file
        self.binary_pituitary_mask = os.path.join(self.output_dir, PITUITARY_MASK_FILE)

        # Convert MNI coordinates to voxel space and get the size as well
        nii_img = nib.load(self.t1_in_mni_space)
        mni_to_voxel = np.linalg.inv(nii_img.affine)
        voxel_coords = np.dot(mni_to_voxel, np.array([*mni_coords[0], 1]))[:3].astype(
            int
        )
        voxel_size = (
            np.dot(mni_to_voxel, np.array([*mni_coords[1], 1]))[:3].astype(int)
            - voxel_coords
        )

        # Set start x, y, z and size x, y, z
        start_x, start_y, start_z = voxel_coords
        size_x, size_y, size_z = voxel_size

        print(
            f"Naive pituitary mask ROI start coordinates: {start_x, start_y, start_z}"
        )
        print(f"Naive pituitary mask ROI sizes: {size_x, size_y, size_z}")

        cmd_mask = [
            "fslmaths",
            self.t1_in_mni_space,
            "-roi",
            str(start_x),
            str(size_x),
            str(start_y),
            str(size_y),
            str(start_z),
            str(size_z),
            "0",
            "1",  # Time dimension
            self.binary_pituitary_mask,
        ]
        subprocess.run(cmd_mask, check=True)

        # Find the peak intensity coordinate within the mask
        cmd_stats = ["fslstats", self.binary_pituitary_mask, "-C"]
        result = subprocess.run(cmd_stats, capture_output=True, text=True, check=True)
        pituitary_voxels = tuple(map(float, result.stdout.strip().split()))

        print(f"Pituitary location (voxel): {pituitary_voxels}")

        # Extract correct slice indices for each orientation
        self.slice_indices = {
            "Axial": voxel_coords[2] + (voxel_size[2] // 2),  # Z-slice
            "Sagittal": voxel_coords[0] + (voxel_size[0] // 2),  # X-slice
            "Coronal": voxel_coords[1] + (voxel_size[1] // 2),  # Y-slice
        }

        # Load the extracted pituitary mask for intensity thresholding
        pituitary_data = nib.load(self.binary_pituitary_mask).get_fdata()
        print(
            f"Pituitary mask intensity range: {np.min(pituitary_data)} - {np.max(pituitary_data)}"
        )
        min_intensity = np.min(
            pituitary_data[pituitary_data > config.intensity_range[0]]
        )  # Ignore background, background tissue, and vessels for now
        max_intensity = np.max(
            pituitary_data[pituitary_data < config.intensity_range[1]]
        )
        highlight_threshold = (min_intensity, max_intensity)

        print(f"Highlighting intensities between {highlight_threshold}")

        # Save the highlighted area as a mask for future reference
        subprocess.run(
            [
                "fslmaths",
                self.binary_pituitary_mask,
                "-thr",
                str(min_intensity),
                "-uthr",
                str(max_intensity),
                "-bin",
                self.binary_pituitary_mask,
            ],
            check=True,
        )

    def __convert_from_mni_to_voxel_space(
        self, mni_coords: Tuple[int, int, int], img: nib.Nifti1Image
    ) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
        """
        Convert MNI coordinates to voxel space using the affine transformation matrix.

        :param mni_coords: A tuple containing MNI coordinates.
        :param img: The T1 image in MNI space.

        :return: A tuple containing the converted voxel coordinates.
        """
        mni_to_voxel = np.linalg.inv(img.affine)

        # Convert MNI coordinates to voxel space
        voxel_coords = np.dot(mni_to_voxel, np.array([*mni_coords, 1]))[:3].astype(int)

        return voxel_coords

    def __create_score_based_mask(
        self,
        centroid: Tuple[int, int, int],
        mni_coords: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
            (config.x_range[0], config.y_range[0], config.z_range[0]),
            (config.x_range[1], config.y_range[1], config.z_range[1]),
        ),  # These coordinates were determined by me
    ) -> np.ndarray:
        """
        Detect and segment the pituitary gland using intensity thresholds and giving preference to
        the naive mask region while considering all voxels within specified MNI coordinates.

        Parameters:
        update_mask (bool): Whether to update the saved pituitary mask file
        mni_coords (tuple): Tuple of two 3D coordinates defining the bounding box in MNI space
        default_centroid (tuple): The default centroid to start the search from

        Returns:
        np.ndarray: The probabilistic pituitary mask
        """
        print("Creating score-based pituitary mask...")

        if not self.binary_pituitary_mask or not os.path.exists(
            self.binary_pituitary_mask
        ):
            raise ValueError(
                "Pituitary mask not found. Run __create_naive_pituitary_mask first."
            )

        # Load the image data
        mask_img = nib.load(self.binary_pituitary_mask)  # Will be the naive mask
        naive_mask_data = mask_img.get_fdata()
        t1_img = nib.load(self.t1_in_mni_space)
        t1_data = t1_img.get_fdata()

        # Convert to voxel space
        voxel_coords_min = self.__convert_from_mni_to_voxel_space(
            mni_coords=mni_coords[0], img=t1_img
        )
        voxel_coords_max = self.__convert_from_mni_to_voxel_space(
            mni_coords=mni_coords[1], img=t1_img
        )

        # Ensure coordinates are ordered properly (min < max)
        x_min, y_min, z_min = np.minimum(voxel_coords_min, voxel_coords_max)
        x_max, y_max, z_max = np.maximum(voxel_coords_min, voxel_coords_max)

        # Ensure coordinates are within image bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        z_min = max(0, z_min)
        x_max = min(t1_data.shape[0] - 1, x_max)
        y_max = min(t1_data.shape[1] - 1, y_max)
        z_max = min(t1_data.shape[2] - 1, z_max)

        # Get coordinates and intensities for all voxels in the specified voxel space region
        config.x_range = range(x_min, x_max + 1)
        config.y_range = range(y_min, y_max + 1)
        config.z_range = range(z_min, z_max + 1)
        coords = (
            np.array(np.meshgrid(config.x_range, config.y_range, config.z_range))
            .reshape(3, -1)
            .T
        )
        intensities = t1_data[coords[:, 0], coords[:, 1], coords[:, 2]]

        def get_connected_component(coords, mask_shape, centroid):
            """
            Get the connected component containing the centroid and high-scoring voxels.

            :param coords: The voxel coordinates
            :param mask_shape: The shape of the mask of high-scoring voxels
            :param centroid: The centroid coordinates

            :return: The connected component mask
            """
            # Create 3D mask
            temp_mask = np.zeros(mask_shape, dtype=bool)
            temp_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True

            # Find connected components
            structure = np.ones((3, 3, 3), dtype=bool)  # 26-connectivity
            labeled_array, num_features = label(temp_mask, structure=structure)

            # Ensure centroid is within bounds
            centroid = np.round(centroid).astype(int)
            if np.all(centroid >= 0) and np.all(centroid < mask_shape):
                if temp_mask[tuple(centroid)]:
                    centroid_label = labeled_array[tuple(centroid)]
                    initial_component = labeled_array == centroid_label

                    # Count neighbors for each voxel in the component
                    neighbor_count = convolve(
                        initial_component.astype(float), structure, mode="constant"
                    )

                    # Only keep voxels with at least high_quality_neighbors_to_consider_connected neighbors
                    connected_mask = (
                        neighbor_count
                        >= config.high_quality_neighbors_to_consider_connected
                    ) & initial_component
                    return connected_mask
            return np.zeros(mask_shape, dtype=bool)

        def compute_scores(
            coords: np.ndarray,
            intensities: np.ndarray,
            centroid: np.ndarray,
            intensity_range: Tuple[float, float],
            mask_shape: Tuple[int, int, int],
        ) -> np.ndarray:
            """
            Calculate clustering scores based on:
            - Distance to centroid
            - Intensity within range
            - Connectivity to centroid

            Weights are used to balance the importance of each score.

            Returns a final score for each voxel.

            :param coords: The voxel coordinates
            :param intensities: The voxel intensities
            :param centroid: The centroid coordinates
            :param intensity_range: The intensity range to consider
            :param mask_shape: The shape of the mask (ideally you want it to be the size of the image so that it overlays correctly)

            :return: The final scores for each voxel
            """
            # Distance score (inverse of distance to centroid)
            distances = np.linalg.norm(coords - centroid, axis=1)
            distance_scores = 1 - (distances / np.max(distances))

            # Intensity score
            min_int, max_int = intensity_range
            intensity_range_size = max_int - min_int
            intensity_scores = np.zeros_like(intensities, dtype=float)

            # Within range gets maximum score
            within_range = (intensities >= min_int) & (intensities <= max_int)
            intensity_scores[within_range] = 1.0

            # Outside range gets decreasing score based on distance from range
            below_range = intensities < min_int
            above_range = intensities > max_int

            intensity_scores[below_range] = 1 - np.minimum(
                1, (min_int - intensities[below_range]) / intensity_range_size
            )
            intensity_scores[above_range] = 1 - np.minimum(
                1, (intensities[above_range] - max_int) / intensity_range_size
            )

            # Connectivity score - strongly prefer voxels connected to centroid
            # First get high-scoring voxels based on other criteria
            initial_scores = (
                (
                    config.distance_weight / config.distance_weight
                    + config.intensity_range_weight
                )
                * distance_scores
            ) + (
                (
                    config.intensity_range_weight / config.distance_weight
                    + config.intensity_range_weight
                )
                * intensity_scores
            )
            high_score_mask = initial_scores >= np.percentile(
                initial_scores, config.min_score_threshold
            )

            # Get connected component from these high-scoring voxels
            connected_mask = get_connected_component(
                coords[high_score_mask], mask_shape, centroid
            )

            # Map back to all coordinates
            connectivity_scores = np.zeros_like(intensities, dtype=float)
            connectivity_scores[high_score_mask] = connected_mask[
                tuple(coords[high_score_mask].T)
            ]

            # Combine scores with weights
            final_scores = (
                config.distance_weight * distance_scores
                + config.intensity_range_weight * intensity_scores
                + config.connectivity_weight * connectivity_scores
            )

            return final_scores

        # Replace the current coordinate transformation code with:
        # Initial MNI coordinates for pituitary
        mni_coords_pituitary = np.array(
            [
                config.default_centroid[0],
                config.default_centroid[1],
                config.default_centroid[2],
            ]
        )  # Adding 1 for homogeneous coordinates
        # Get voxel coordinates and ensure they are ints
        voxel_coords = self.__convert_from_mni_to_voxel_space(
            mni_coords=mni_coords_pituitary, img=t1_img
        )
        centroid = np.round(voxel_coords).astype(int)

        # Save centroid as nii.gz file for visualization
        centroid_mask = np.zeros_like(naive_mask_data)
        centroid_mask[
            np.round(centroid[0]).astype(int),
            np.round(centroid[1]).astype(int),
            np.round(centroid[2]).astype(int),
        ] = 1

        centroid_img = nib.Nifti1Image(centroid_mask, mask_img.affine, mask_img.header)
        self.centroid_mask = os.path.join(self.output_dir, PITUITARY_CENTROID_FILE)
        nib.save(centroid_img, self.centroid_mask)

        # Calculate clustering scores
        scores = compute_scores(
            coords, intensities, centroid, config.intensity_range, t1_data.shape
        )

        # Instead of creating a binary mask, create a probabilistic one using the scores
        prob_mask = np.zeros_like(naive_mask_data)
        prob_mask[coords[:, 0], coords[:, 1], coords[:, 2]] = scores

        # Apply threshold but keep probabilities
        prob_mask[prob_mask < config.min_score_threshold] = 0

        print(f"Score-based pituitary mask created with {np.sum(prob_mask)} voxels")

        # Create a scores visualization volume
        scores_volume = np.zeros_like(naive_mask_data)
        scores_volume[coords[:, 0], coords[:, 1], coords[:, 2]] = scores

        # Save scores as a NIfTI file for visualization
        scores_img = nib.Nifti1Image(scores_volume, mask_img.affine, mask_img.header)
        self.score_based_mask_scores = os.path.join(
            self.output_dir, PROB_PITUITARY_MASK
        )
        nib.save(scores_img, self.score_based_mask_scores)

        # Show the original visualization with an additional scores view
        show_mri_slices(
            [
                self.t1_in_mni_space,
                self.centroid_mask,
                self.score_based_mask_scores,
            ],
            slice_index=self.slice_indices,
            titles=["T1w MNI w/ Score-Based Mask"],
            overlay=True,
            colormaps=[
                "gray",  # T1 image in grayscale
                "hot",  # Centroid in hot colors
                "viridis",  # Scores in viridis colormap
            ],
        )

        # Return selected coordinates and their probabilities instead of binary mask
        return prob_mask

    def __create_probabilistic_pituitary_mask(
        self,
        centroid: Tuple[int, int, int],
        mni_coords: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
            (config.x_range[0], config.y_range[0], config.z_range[0]),
            (config.x_range[1], config.y_range[1], config.z_range[1]),
        ),
    ) -> np.ndarray:
        """
        Create a probabilistic pituitary mask using both score-based masking and region-growing methods.

        Parameters:
        mni_coords: Tuple of boundary coordinates in MNI space
        centroid: the centroid we are basing our masking around

        Returns:
        np.ndarray: Probabilistic mask where values represent confidence (0-1)
        """
        print("Creating probabilistic pituitary mask using combined methods...")

        # Create naive mask
        self.__create_naive_pituitary_mask(mni_coords)

        # Get the score-based mask scores
        score_based_probs = self.__create_score_based_mask(
            mni_coords=mni_coords,
            centroid=centroid,
        )

        # Create probabilistic mask
        prob_mask = np.zeros_like(score_based_probs)

        # Get the region growing mask
        t1_img = nib.load(self.t1_in_mni_space)
        t1_data = t1_img.get_fdata()

        # Get centroid in voxel space
        centroid = self.__convert_from_mni_to_voxel_space(centroid, t1_img)

        # Cap probabilities at 1.0
        prob_mask = np.minimum(prob_mask, 1.0)

        # Check if the centroid is within the intensity range. If not then use the centroid of the score-based mask
        if (
            t1_data[tuple(centroid)] < config.intensity_range[0]
            or t1_data[tuple(centroid)] > config.intensity_range[1]
        ):
            print(
                f"WARNING: Centroid {centroid} was not in the intensity range skipping region growing"
            )

            # Try to find a better centroid by finding the nearest voxel within the intensity range
            centroid = np.unravel_index(
                np.argmax(
                    score_based_probs
                    * (t1_data >= config.intensity_range[0])
                    * (t1_data <= config.intensity_range[1])
                ),
                score_based_probs.shape,
            )

            # See if any centroid was found
            if (
                t1_data[centroid] < config.intensity_range[0]
                or t1_data[centroid] > config.intensity_range[1]
            ):
                # Skip region growing if no centroid was found
                print("No centroid found in intensity range. Skipping region growing.")
                return score_based_probs

            print(f"New centroid found: {centroid}")

        region_mask = self.__region_growing(t1_data, tuple(centroid))

        # Show the region_mask
        region_mask_img = nib.Nifti1Image(region_mask, t1_img.affine)
        self.region_mask = os.path.join(
            self.output_dir, REGION_GROWING_MASK_PREFIX + PITUITARY_MASK_FILE
        )
        nib.save(region_mask_img, self.region_mask)

        # Show the region mask overlayed on the T1 image
        show_mri_slices(
            [self.t1_in_mni_space, self.region_mask],
            slice_index=self.slice_indices,
            titles=["T1 MRI with Region Growing Mask Overlay"],
            overlay=True,
            colormaps=["gray", "viridis"],  # T1 image  # Region mask
        )

        # Assign probabilities based on voting
        # Combine probabilities (0.6 weight for score-based, 0.4 for region growing)
        prob_mask = config.score_based_weight * score_based_probs
        prob_mask[region_mask > 0] += config.region_growing_weight

        # Neighbors-based boosting
        # Fill in voxels surrounded by high-probability neighbors
        structure = np.ones((3, 3, 3))  # 26-connectivity neighborhood
        for _ in range(2):  # Do this twice to ensure good coverage
            # Find voxels with high probability neighbors
            neighbor_sum = convolve(
                (prob_mask > config.min_score_considered_high_score).astype(
                    float
                ),  # Look at very high probability voxels
                structure,
                mode="constant",
            )
            # If a voxel has 20+ high probability neighbors (out of 26 possible), make it high probability
            high_neighbor_mask = (
                (neighbor_sum > config.num_neighbors_required_to_boost)
                & (prob_mask < config.min_score_considered_high_score)
                & (prob_mask > config.min_score_to_boost_if_quality_neighbors)
            )
            prob_mask[high_neighbor_mask] = config.min_score_considered_high_score

        print("Created probabilistic mask. Returning refined mask.")

        return prob_mask

    def __remove_appendages(
        self, prob_mask: np.ndarray, t1_img_affine: np.ndarray, keep_inf_extension=True
    ):
        """
        Removes appendages from the probabilistic pituitary mask while allowing
        the vertical infundibulum extension.

        Parameters:
        prob_mask (np.ndarray): The probabilistic mask.
        t1_img_affine (np.ndarray): Affine transformation of the T1 image.
        keep_inf_extension (bool): Whether to preserve the vertical infundibulum.

        Returns:
        np.ndarray: The cleaned mask.
        """
        print("Removing appendages...")

        # Threshold mask (convert to binary)
        bin_mask = (
            prob_mask > config.final_score_threshold
        )  # Keeping high-confidence voxels

        # Label connected components
        labeled_mask, num_features = label(bin_mask)

        # Compute sizes of components
        component_sizes = np.bincount(labeled_mask.ravel())
        component_sizes[0] = 0  # Ignore background (label 0)

        # Keep the largest connected component (assumed to be the main pituitary gland)
        largest_component = np.argmax(component_sizes)

        # Create cleaned mask
        cleaned_mask = np.zeros_like(prob_mask)
        cleaned_mask[labeled_mask == largest_component] = prob_mask[
            labeled_mask == largest_component
        ]

        if keep_inf_extension:
            # Convert MNI [0, 0, -20] (infundibulum region) to voxel space
            mni_inf_position = np.array(
                [0, 0, max(config.z_range), 1]
            )  # Adjust z if needed
            voxel_inf_position = np.dot(np.linalg.inv(t1_img_affine), mni_inf_position)[
                :3
            ]
            voxel_inf_position = np.round(voxel_inf_position).astype(int)

            # Find the component that extends upward near MNI x ≈ 0
            for i in range(1, num_features + 1):
                coords = np.argwhere(labeled_mask == i)
                if np.any(
                    np.abs(coords[:, 0] - voxel_inf_position[0])
                    < config.infundibulum_range
                ):  # Allow small x variation
                    cleaned_mask[labeled_mask == i] = prob_mask[labeled_mask == i]

        # Apply morphological closing to remove small gaps and smooth boundaries
        cleaned_mask = binary_closing(
            cleaned_mask, ball(config.appendage_removal_radius)
        ).astype(np.uint8)

        print("Appendages removed. Returning refined mask.")
        return cleaned_mask

    def __region_growing(
        self,
        image: np.ndarray,
        seed: Tuple[int, int, int],
        intensity_tol: int = config.intensity_tolerance,
        max_voxels: int = config.max_voxels,
    ) -> np.ndarray:
        """
        Perform region growing segmentation from the given seed point.

        :param image: 3D NumPy array of the T1 MRI
        :param seed: Tuple (x, y, z) representing the centroid
        :param intensity_tol: Allowed intensity variation for region growing
        :param max_voxels: Upper limit for segmented region size (to prevent overgrowth)

        :return: Binary mask of the segmented region
        """
        x, y, z = seed
        seed_intensity = image[x, y, z]

        mask = np.zeros_like(image, dtype=np.uint8)
        mask[x, y, z] = 1  # Mark seed point as segmented

        queue = [(x, y, z)]
        count = 1

        while queue and count < max_voxels:
            cx, cy, cz = queue.pop(0)

            # Check 6-connected neighbors
            for dx, dy, dz in [
                (-1, 0, 0),
                (1, 0, 0),
                (0, -1, 0),
                (0, 1, 0),
                (0, 0, -1),
                (0, 0, 1),
            ]:
                nx, ny, nz = cx + dx, cy + dy, cz + dz

                if (
                    0 <= nx < image.shape[0]
                    and 0 <= ny < image.shape[1]
                    and 0 <= nz < image.shape[2]
                    and mask[nx, ny, nz] == 0
                    and abs(image[nx, ny, nz] - seed_intensity) < intensity_tol
                ):

                    mask[nx, ny, nz] = 1
                    queue.append((nx, ny, nz))
                    count += 1

        # Save the region growing mask
        region_growing_mask_image = nib.Nifti1Image(
            mask, nib.load(self.t1_in_mni_space).affine
        )
        self.region_growing_mask = os.path.join(
            self.output_dir, f"{REGION_GROWING_MASK_PREFIX}{PITUITARY_MASK_FILE}"
        )
        nib.save(region_growing_mask_image, self.region_growing_mask)

        return mask

    def __get_pituitary_statistics(self) -> dict:
        """
        Calculate statistics about the detected pituitary region

        Returns:
        dict: Dictionary containing various statistics about the pituitary region
        """
        print("Calculating pituitary statistics...")

        if not self.binary_pituitary_mask or not os.path.exists(
            self.binary_pituitary_mask
        ):
            raise ValueError(
                "Pituitary mask not found. Run __create_naive_pituitary_mask first."
            )

        mask_img = nib.load(self.binary_pituitary_mask)
        mask_data = mask_img.get_fdata()
        t1_img = nib.load(self.t1_in_mni_space)
        t1_data = t1_img.get_fdata()

        # Get coordinates of mask voxels
        mask_coords = np.array(np.where(mask_data > 0)).T

        if len(mask_coords) == 0:
            print("No voxels found in pituitary mask.")
            # raise ValueError("No voxels found in pituitary mask")

        # Calculate center of mass in voxel space
        # Replace the center of mass calculation with:
        mask_coords = np.array(np.where(mask_data > 0)).T
        voxel_probabilities = mask_data[
            mask_data > 0
        ]  # Get probability scores for each voxel

        # Calculate weighted center of mass using probability scores as weights
        # If voxel probabilities sum to 0, then keep center of mass and mni center the same
        center_of_mass = np.average(mask_coords, weights=voxel_probabilities, axis=0)

        # Convert to MNI space
        mni_center = np.dot(mask_img.affine, np.append(center_of_mass, 1))[:3]

        # Get intensity statistics
        mask_intensities = t1_data[
            mask_coords[:, 0], mask_coords[:, 1], mask_coords[:, 2]
        ]

        self.final_mask_stats = {
            "volume_voxels": len(mask_coords),
            "volume_mm3": len(mask_coords)
            * np.prod(np.abs(np.diag(mask_img.affine)[:3])),
            "center_mni": mni_center,
            "mean_intensity": np.mean(mask_intensities),
            "min_intensity": np.min(mask_intensities),
            "max_intensity": np.max(mask_intensities),
            "std_intensity": np.std(mask_intensities),
        }

        print("Completed pituitary statistics calculation:")
        for key, value in self.final_mask_stats.items():
            print(f"{key}: {value}")

        return self.final_mask_stats

    def __create_output_directory(self) -> None:
        """
        Create output directory structure. Uses the subject ID to create a directory structure for processed data.
        The root directory and suffix added after the subject ID are defined in the constants file.

        The structure is as follows:
        {PATH_TO_PROCESSED_DATA}/{subject_id}{SUFFIX_SUBJECT_ID_PROCESSED}/{subject_id}/

        :return: None. Sets the output directory path as an instance variable of the class. Called output_dir
        """
        output_dir = os.path.join(
            PATH_TO_PROCESSED_DATA,
            f"{self.subject_id}{SUFFIX_SUBJECT_ID_PROCESSED}",
            self.subject_id,
        )
        os.makedirs(output_dir, exist_ok=True)

        self.output_dir = output_dir

    def __motion_correction_and_smoothing(self) -> None:
        """
        Perform motion correction and smoothing using mcflirt and fslmaths.

        :return: None. Sets the motion-corrected and smoothed MRI paths as instance variables. These are:
            motion_corrected_t1w, motion_corrected_t2w, smoothed_and_mc_t1w, smoothed_and_mc_t2w
        """
        print("Correcting for motion artifact & smoothing...")

        # Paths to motion corrected
        self.motion_corrected_t1w = os.path.join(self.output_dir, T1_MC_FILE)
        self.motion_corrected_t2w = os.path.join(self.output_dir, T2_MC_FILE)

        # Motion correction using mcflirt
        subprocess.run(
            ["mcflirt", "-in", self.unprocessed_t1, "-out", self.motion_corrected_t1w],
            check=True,
        )
        subprocess.run(
            ["mcflirt", "-in", self.unprocessed_t2, "-out", self.motion_corrected_t2w],
            check=True,
        )

        # Show motion correction
        show_mri_slices(
            [self.motion_corrected_t1w, self.motion_corrected_t2w],
            titles=["T1w Motion Corrected MRI", "T2w Motion Corrected MRI"],
        )

        # Apply Gaussian smoothing

        # Paths to smoothed and motion corrected
        self.smoothed_and_mc_t1w = os.path.join(self.output_dir, T1_SMOOTH_FILE)
        self.smoothed_and_mc_t2w = os.path.join(self.output_dir, T2_SMOOTH_FILE)
        subprocess.run(
            [
                "fslmaths",
                self.motion_corrected_t1w,
                "-s",
                str(config.smoothing_sigma),
                self.smoothed_and_mc_t1w,
            ],
            check=True,
        )
        subprocess.run(
            [
                "fslmaths",
                self.motion_corrected_t2w,
                "-s",
                str(config.smoothing_sigma),
                self.smoothed_and_mc_t2w,
            ],
            check=True,
        )

        # Show smoothed images
        show_mri_slices(
            [self.smoothed_and_mc_t1w, self.smoothed_and_mc_t2w],
            titles=["T1w Smoothed MRI", "T2w Smoothed MRI"],
        )

    def __registration_and_normalization(self, use_nonlinear_reg: bool = False) -> None:
        """
        Perform registration and normalization using FLIRT and FNIRT.
        This function completes the following steps:
            1. Affine registration of the T1w MRI to the T2w MRI using FLIRT.
            2. Apply the affine transformation to the T1w MRI to align it with the T2w MRI.
            3. Display the results of the registration.

        :param use_nonlinear_reg: Whether to use FNIRT for registration. Default is False.

        :return: None. Sets the paths to the registered and normalized T1w MRI as instance variables. These are:
            t1w_smooth_reg_t2, affine_matrix, t1w_preproc_reg_t2
        """
        print("Completing registration & normalization using FLIRT...")

        # Paths
        self.t1w_smooth_reg_t2 = os.path.join(
            self.output_dir, T1_SMOOTH_REGISTERED_FILE
        )
        self.affine_matrix = os.path.join(self.output_dir, T1_TO_T2_MAT_FILE)
        self.t1w_preproc_reg_t2 = os.path.join(self.output_dir, T1_MC_REGISTERED_FILE)

        # FLIRT (Affine Registration)
        subprocess.run(
            [
                "flirt",
                "-in",
                self.smoothed_and_mc_t1w,
                "-ref",
                self.smoothed_and_mc_t2w,
                "-out",
                self.t1w_smooth_reg_t2,
                "-omat",
                self.affine_matrix,
            ],
            check=True,
        )

        # Apply the affine transformation to the T1 image (output is already aligned to T2)
        subprocess.run(
            [
                "flirt",
                "-in",
                self.motion_corrected_t1w,
                "-ref",
                self.motion_corrected_t2w,
                "-out",
                self.t1w_preproc_reg_t2,
                "-applyxfm",  # Use affine transformation matrix to apply the registration
                "-init",
                self.affine_matrix,
            ],
            check=True,
        )

        print("Affine Normalization Completed. T1 MRI in T2 Space.")

        if use_nonlinear_reg:
            # Path for warp field
            self.t1_to_t2_warp = os.path.join(self.output_dir, T1_TO_T2_WARP_FILE)
            if os.path.exists(self.warp_field):
                print(
                    "Starting Nonlinear Registration using FNIRT for T1 to T2 space..."
                )

                # Compute Nonlinear Warp using FNIRT
                subprocess.run(
                    [
                        "fnirt",
                        f"--in={self.smoothed_and_mc_t1w}",
                        f"--ref={self.smoothed_and_mc_t2w}",
                        f"--aff={self.affine_matrix}",
                        f"--cout={self.t1_to_t2_warp}",
                        f"--iout={self.t1w_smooth_reg_t2}",
                        f"--splineorder={str(config.spline_order)}",
                        f"--numprec={config.hessian_precision}",
                        "--verbose",
                    ],
                    check=True,
                )

                print(
                    f"Nonlinear registration (FNIRT) of T1 to T2 completed. Warp file is saved at {self.t1_to_t2_warp}"
                )

            # Step 3: Apply the Transformation to Non-blurry Motion-Corrected Images

            print(
                "Applying warp transformations to motion-corrected images for T1 image..."
            )

            # Apply affine + nonlinear warp to motion-corrected T1
            subprocess.run(
                [
                    "applywarp",
                    "-i",
                    self.motion_corrected_t1w,  # Non-smoothed motion-corrected T1
                    "-r",
                    self.smoothed_and_mc_t2w,
                    "-w",
                    self.t1_to_t2_warp,
                    "-o",
                    self.t1w_preproc_reg_t2,
                ],
                check=True,
            )

            print("Nonlinear Registration Completed. T1 MRI in T2 Space.")

        # Display results
        show_mri_slices(
            [self.t1w_preproc_reg_t2, self.motion_corrected_t2w],
            titles=["T1w Affine Normalized MRI", "T2w Motion Corrected MRI"],
        )

    def __delete_temp_files(
        self,
        step_completed: Literal["preprocessed", "in_mni", "pituitary_segmented"],
    ) -> None:
        """
        Deletes temporary files created during the preprocessing pipeline.

        :param step_completed: The step in the pipeline to delete temporary files for.

        :return: None
        """
        print("Deleting temporary files...")

        if step_completed not in ["preprocessed", "in_mni", "pituitary_segmented"]:
            raise ValueError(
                "Step completed must be 'preprocessed', 'in_mni', or 'pituitary_segmented'"
            )

        # All steps will have these file no longer needed
        (
            os.remove(self.smoothed_and_mc_t1w)
            if os.path.exists(self.smoothed_and_mc_t1w)
            else None
        )
        (
            os.remove(self.smoothed_and_mc_t2w)
            if os.path.exists(self.smoothed_and_mc_t2w)
            else None
        )

        if step_completed == "in_mni":
            (
                os.remove(self.t1w_smooth_reg_t2)
                if os.path.exists(self.t1w_smooth_reg_t2)
                else None
            )
            (
                os.remove(self.t1w_preproc_reg_t2)
                if os.path.exists(self.t1w_preproc_reg_t2)
                else None
            )
            (
                os.remove(self.motion_corrected_t1w)
                if os.path.exists(self.motion_corrected_t1w)
                else None
            )
            (
                os.remove(self.motion_corrected_t2w)
                if os.path.exists(self.motion_corrected_t2w)
                else None
            )
            (
                os.remove(self.affine_matrix)
                if os.path.exists(self.affine_matrix)
                else None
            )
