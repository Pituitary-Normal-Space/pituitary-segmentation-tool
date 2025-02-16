"""
File that contains the Subject class. This is a class that represents a subject, their details, and stores their MRI data.
"""

# Standard libs
import os
import subprocess
from typing import Literal, Tuple, Dict

# Non-standard libs
import numpy as np
import nibabel as nib
from scipy.ndimage import label

# sfrom sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


# Local libs
from config import (
    smoothing_sigma,
    fractional_intensity_t1,
    fractional_intensity_t2,
    gradient_t1,
    gradient_t2,
    robust_brain_extraction,
    spline_order,
    hessian_precision,
    x_range,
    y_range,
    z_range,
    intensity_range,
    distance_weight,
    intensity_range_weight,
    connectivity_weight,
    naive_mask_weight,
    min_score_threshold,
    cluster_dist_threshold,
)
from const import (
    T1_BRAIN_FILE,
    T2_BRAIN_FILE,
    T1_MC_FILE,
    T2_MC_FILE,
    T1_SMOOTH_FILE,
    T2_SMOOTH_FILE,
    T1_SMOOTH_REGISTERED_FILE,
    T1_TO_T2_MAT_FILE,
    T1_MC_REGISTERED_FILE,
    T1_AFFINE_TO_MNI_FILE,
    T1_SMOOTH_NONLIN_TO_MNI_FILE,
    T1_WARP_FILE,
    T1_MNI_FILE,
    T2_MNI_FILE,
    OVERLAY_FILE,
    PITUITARY_MASK_FILE,
    T1_MNI_MAT_FILE,
    PATH_TO_PROCESSED_DATA,
    MNI_TEMPLATE,
    PITUITARY_CENTROID_FILE,
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
    """

    def __init__(
        self,
        subject_id: str,
        age: Literal["31-35", "26-30", "22-25"],
        sex: Literal["M", "F"],
        t1_path: str,
        t2_path: str,
    ):
        self.subject_id = subject_id
        self.age = age
        self.sex = sex
        self.unprocessed_t1 = t1_path
        self.unprocessed_t2 = t2_path
        self.processeing_complete = False
        self.processed_t1 = None
        self.processed_t2 = None
        self.output_dir = None
        self.motion_corrected_t1w = None
        self.motion_corrected_t2w = None
        self.smoothed_t1w = None
        self.smoothed_t2w = None
        self.registered_t1w = None
        self.affine_matrix = None
        self.normalized_t1w = None
        self.warp_field = None
        self.final_t1_mni = None
        self.final_t2_mni = None
        self.overlay_output = None
        self.in_MNI_space = False
        self.pituitary_mask = None
        self.final_mask_stats = None

        # Check that config parameters are valid
        if min_score_threshold < 0 or min_score_threshold > 1:
            raise ValueError("Minimum score threshold must be between 0 and 1")

        if (
            round(
                distance_weight
                + intensity_range_weight
                + connectivity_weight
                + naive_mask_weight,
                10,
            )
            != 1
        ):
            raise ValueError(
                f"Weights for distance, intensity, connectivity, and naive mask must add up to 1 they are {distance_weight}, {intensity_range_weight}, {connectivity_weight}, {naive_mask_weight} adding to {distance_weight + intensity_range_weight + connectivity_weight + naive_mask_weight}"
            )

        if fractional_intensity_t1 < 0 or fractional_intensity_t1 > 1:
            raise ValueError(
                "Fractional intensity threshold for T1 must be between 0 and 1"
            )

        if fractional_intensity_t2 < 0 or fractional_intensity_t2 > 1:
            raise ValueError(
                "Fractional intensity threshold for T2 must be between 0 and 1"
            )

        if gradient_t1 < 0:
            raise ValueError("Gradient threshold for T1 must be greater than 0")

        if gradient_t2 < 0:
            raise ValueError("Gradient threshold for T2 must be greater than 0")

        if smoothing_sigma < 0:
            raise ValueError("Smoothing sigma must be greater than 0")

        if spline_order not in [2, 3]:
            raise ValueError("Spline order must be 2 or 3")

        if hessian_precision not in ["double", "float"]:
            raise ValueError("Hessian precision must be double or float")

    def __str__(self) -> str:
        base_info = (
            f"Subject ID: {self.subject_id}\n"
            f"Age: {self.age}\n"
            f"Sex: {self.sex}\n"
            f"Processing Complete: {self.processeing_complete}\n"
            f"UnProcessed T1 Path: {self.unprocessed_t1}\n"
            f"UnProcessed T2 Path: {self.unprocessed_t2}"
        )

        if not self.processeing_complete:
            return base_info

        processed_info = (
            f"\nProcessed T1 MNI Path: {self.final_t1_mni}\n"
            f"Processed T2 MNI Path: {self.final_t2_mni}"
        )

        if self.pituitary_mask and self.final_mask_stats:
            mask_info = (
                f"\nPituitary Mask Path: {self.pituitary_mask}\n"
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
        self.__create_output_directory()
        self.processeing_complete = True

        self.motion_corrected_t1w = os.path.join(self.output_dir, T1_MC_FILE)
        self.motion_corrected_t2w = os.path.join(self.output_dir, T2_MC_FILE)
        self.smoothed_t1w = os.path.join(self.output_dir, T1_SMOOTH_FILE)
        self.smoothed_t2w = os.path.join(self.output_dir, T2_SMOOTH_FILE)
        self.registered_t1w = os.path.join(self.output_dir, T1_SMOOTH_REGISTERED_FILE)
        self.affine_matrix = os.path.join(self.output_dir, T1_TO_T2_MAT_FILE)
        self.normalized_t1w = os.path.join(self.output_dir, T1_MC_REGISTERED_FILE)
        self.final_t1_mni = os.path.join(self.output_dir, T1_MNI_FILE)
        self.final_t2_mni = os.path.join(self.output_dir, T2_MNI_FILE)
        self.overlay_output = os.path.join(self.output_dir, OVERLAY_FILE)

    def preprocess_MRIs(self) -> None:
        """
        Preprocesses T1w_MPR and T2w_SPC MRIs using:
        - FAST (FSL’s automated segmentation tool) for tissue segmentation (gray matter, white matter, CSF).
        - FEAT (FSL’s motion correction & smoothing) for motion correction & Gaussian smoothing.
        - FLIRT (FSL’s linear registration) for affine alignment.
        - FNIRT (FSL’s nonlinear registration) for fine-grained normalization.
        The final processed images will be stored in a structured output directory.
        """
        self.__create_output_directory()

        # Step 1: Tissue segmentation using FAST (instead of BET)
        # Having trouble with this and it's not necessary for the project potentially since we need the pituitary anyways
        # self.__perform_brain_extraction()

        # Step 2: Motion Correction & Smoothing using FEAT
        self.__motion_correction_and_smoothing()

        # Step 3: Registration & Normalization using FLIRT and FNIRT
        self.__registration_and_normalization()

        print(
            "Preprocessing completed successfully. This means that the T1W image is in T2W image space and both are motion corrected. Outputs stored in:",
            self.output_dir,
        )

        self.processeing_complete = True

    def overlay_MRIs(self) -> None:
        """
        Function to overlay the T1 and T2 MRI data of the subject.
        This step is performed **after** preprocessing is complete.
        The overlay allows for a visual comparison between the two modalities.
        """

        # Construct file paths for the preprocessed images
        self.overlay_output = os.path.join(self.output_dir, OVERLAY_FILE)

        # Ensure the preprocessed images exist before overlaying
        if not os.path.exists(self.registered_t1w) or not os.path.exists(
            self.motion_corrected_t2w
        ):
            raise FileNotFoundError(
                "Preprocessed images not found. Run preprocess_MRIs first."
            )

        # Overlay T1w and T2w images using FSL’s fslmaths
        subprocess.run(
            [
                "fslmaths",
                self.registered_t1w,
                "-add",
                self.motion_corrected_t2w,
                self.overlay_output,
            ],
            check=True,
        )

        # Show a slice of the overlayed image
        show_mri_slices([self.overlay_output], titles=["T1w T2w Overlay MRI"])

        print(f"Overlaying MRIs completed. Output saved at {self.overlay_output}.")

    def coregister_to_mni(
        self,
        mni_template_path: str = MNI_TEMPLATE,
    ) -> None:
        """
        Coregisters both the T1 and T2 images to the MNI template using affine and nonlinear warping.

        The transformation is computed on the smoothed T1 image, but the transformation is later
        applied to the non-smoothed, motion-corrected T1 and T2 images to avoid excessive blurring.
        """
        # Paths to save transformations
        self.affine_matrix = os.path.join(self.output_dir, T1_MNI_MAT_FILE)
        self.warp_field = os.path.join(self.output_dir, T1_WARP_FILE)

        # Output paths for final non-blurry normalized images
        self.final_t1_mni = os.path.join(self.output_dir, T1_MNI_FILE)
        self.final_t2_mni = os.path.join(self.output_dir, T2_MNI_FILE)

        # If not previously coregistered and warp field exists, skip registration
        if not os.path.exists(self.warp_field):
            print(
                "Transforming to MNI space using smoothed T1 image... Starting with affine linear registration."
            )

            # Compute Affine Transformation using FLIRT
            subprocess.run(
                [
                    "flirt",
                    "-in",
                    self.smoothed_t1w,  # Smoothed T1 used for registration
                    "-ref",
                    mni_template_path,
                    "-out",
                    os.path.join(self.output_dir, T1_AFFINE_TO_MNI_FILE),
                    "-omat",
                    self.affine_matrix,
                ],
                check=True,
            )

            print(
                "Affine registration to MNI completed. Starting non-linear registration. This may take a while..."
            )

            # Compute Nonlinear Warp using FNIRT
            subprocess.run(
                [
                    "fnirt",
                    f"--in={self.smoothed_t1w}",
                    f"--ref={mni_template_path}",
                    f"--aff={self.affine_matrix}",
                    f"--cout={self.warp_field}",
                    f"--iout={os.path.join(self.output_dir, T1_SMOOTH_NONLIN_TO_MNI_FILE)}",
                    f"--splineorder={str(spline_order)}",
                    f"--numprec={hessian_precision}",
                    "--verbose",
                ],
                check=True,
            )

            print(
                f"Nonlinear registration (FNIRT) to MNI completed. Warp file is saved at {self.warp_field}"
            )

        # Step 3: Apply the Transformation to Non-blurry Motion-Corrected Images

        print(
            "Applying warp transformations to motion-corrected images for T1 and T2 images..."
        )

        # Apply affine + nonlinear warp to motion-corrected T1
        subprocess.run(
            [
                "applywarp",
                "-i",
                self.normalized_t1w,  # Non-smoothed motion-corrected T1
                "-r",
                mni_template_path,
                "-w",
                self.warp_field,
                "-o",
                self.final_t1_mni,
            ],
            check=True,
        )

        print(
            f"T1 Motion Corrected Image Registered to MNI. Located here: {self.final_t1_mni}"
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
                self.final_t2_mni,
            ],
            check=True,
        )

        print(
            f"T2 Motion Corrected Image Registered to MNI. Located here: {self.final_t2_mni}"
        )

        # Show results
        show_mri_slices(
            [self.final_t1_mni, self.final_t2_mni, mni_template_path],
            titles=[
                "T1w Motion Corrected Normalized to MNI",
                "T2w Motion Corrected Normalized to MNI",
                "MNI Template",
            ],
        )

        self.in_MNI_space = True

    def complete_pituitary_analysis(
        self,
        mni_coords: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
            (x_range[0], y_range[0], z_range[0]),
            (x_range[1], y_range[1], z_range[1]),
        ),
    ) -> Dict[str, float]:
        """
        Creates a probabilistic pituitary mask using both methods and calculates statistics.

        Returns:
        Dict[str, float]: Statistics about the detected pituitary region
        """
        self.__create_naive_pituitary_mask(mni_coords)
        prob_mask = self.__create_probabilistic_pituitary_mask(mni_coords)

        # Convert probabilistic mask to binary for statistics using 0.5 threshold
        binary_mask_img = nib.Nifti1Image(
            (prob_mask > 0.5).astype(np.uint8), nib.load(self.final_t1_mni).affine
        )
        self.pituitary_mask = os.path.join(self.output_dir, PITUITARY_MASK_FILE)
        nib.save(binary_mask_img, self.pituitary_mask)

        self.__get_pituitary_statistics()
        return self.final_mask_stats

    def __create_naive_pituitary_mask(
        self,
        mni_coords: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
            (x_range[0], y_range[0], z_range[0]),
            (x_range[1], y_range[1], z_range[1]),
        ),  # These coordinates were determined by me
    ) -> Tuple[float, float, float]:
        """
        Locate the pituitary gland within the processed T1-weighted MRI scan using FSL tools.

        :param mni_coords: A tuple containing two MNI coordinate bounds to define the search region.
        :return: The MNI coordinates of the detected pituitary gland location.
        """
        print("Creating naive pituitary mask...")
        if not self.final_t1_mni:
            raise FileNotFoundError(
                "T1 MRI image in MNI space not found. Run coregister_to_mni first."
            )

        # Define the output mask file
        self.pituitary_mask = os.path.join(self.output_dir, PITUITARY_MASK_FILE)

        # Convert MNI coordinates to voxel space and get the size as well
        nii_img = nib.load(self.final_t1_mni)
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
            self.final_t1_mni,
            "-roi",
            str(start_x),
            str(size_x),
            str(start_y),
            str(size_y),
            str(start_z),
            str(size_z),
            "0",
            "1",  # Time dimension
            self.pituitary_mask,
        ]
        subprocess.run(cmd_mask, check=True)

        # Find the peak intensity coordinate within the mask
        cmd_stats = ["fslstats", self.pituitary_mask, "-C"]
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
        pituitary_data = nib.load(self.pituitary_mask).get_fdata()
        print(
            f"Pituitary mask intensity range: {np.min(pituitary_data)} - {np.max(pituitary_data)}"
        )
        min_intensity = np.min(
            pituitary_data[pituitary_data > intensity_range[0]]
        )  # Ignore background, background tissue, and vessels for now
        max_intensity = np.max(pituitary_data[pituitary_data < intensity_range[1]])
        highlight_threshold = (min_intensity, max_intensity)

        print(f"Highlighting intensities between {highlight_threshold}")

        # Visualize the detected region
        show_mri_slices(
            [self.pituitary_mask],
            slice_index=self.slice_indices,  # Pass all three slice indices
            titles=["Pituitary Region With Highlights"],
            highlight_threshold=highlight_threshold,
        )

        # Save the highlighted area as a mask for future reference
        subprocess.run(
            [
                "fslmaths",
                self.pituitary_mask,
                "-thr",
                str(min_intensity),
                "-uthr",
                str(max_intensity),
                "-bin",
                self.pituitary_mask,
            ],
            check=True,
        )

        return pituitary_voxels

    def __create_dynamic_pituitary_mask(
        self,
        update_mask: bool = True,
        mni_coords: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
            (x_range[0], y_range[0], z_range[0]),
            (x_range[1], y_range[1], z_range[1]),
        ),  # These coordinates were determined by me
        dynamic_centroid: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect and segment the pituitary gland using intensity thresholds and giving preference to
        the naive mask region while considering all voxels within specified MNI coordinates.

        Parameters:
        update_mask (bool): Whether to update the saved pituitary mask file
        mni_coords (tuple): Tuple of two 3D coordinates defining the bounding box in MNI space
        dynamic_centroid (bool): Whether to use the naive mask centroid as the clustering centroid or use hard-coded values of the pituitary region
            Hard coded values are (0, 2, -32) in MNI space.

        Returns:
        tuple: (coordinates of selected voxels, final binary mask)
        """
        print("Creating dynamic pituitary mask...")

        if not self.pituitary_mask or not os.path.exists(self.pituitary_mask):
            raise ValueError(
                "Pituitary mask not found. Run __create_naive_pituitary_mask first."
            )

        # Load the mask and image data
        mask_img = nib.load(self.pituitary_mask)
        naive_mask_data = mask_img.get_fdata()
        t1_img = nib.load(self.final_t1_mni)
        t1_data = t1_img.get_fdata()

        # Convert MNI coordinates to voxel space
        mni_to_voxel = np.linalg.inv(t1_img.affine)

        # Convert both min and max coordinates
        voxel_coords_min = np.dot(mni_to_voxel, np.array([*mni_coords[0], 1]))[
            :3
        ].astype(int)
        voxel_coords_max = np.dot(mni_to_voxel, np.array([*mni_coords[1], 1]))[
            :3
        ].astype(int)

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
        x_range = range(x_min, x_max + 1)
        y_range = range(y_min, y_max + 1)
        z_range = range(z_min, z_max + 1)
        coords = np.array(np.meshgrid(x_range, y_range, z_range)).reshape(3, -1).T
        intensities = t1_data[coords[:, 0], coords[:, 1], coords[:, 2]]

        def get_connected_component(coords, mask_shape, centroid):
            """Get the connected component containing the centroid"""
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
                    connected_mask = labeled_array == centroid_label
                    return connected_mask
            return np.zeros(mask_shape, dtype=bool)

        def calculate_clustering_scores(
            coords, intensities, centroid, intensity_range, naive_mask, mask_shape
        ):
            """Calculate clustering scores based on multiple criteria"""
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

            # Naive mask presence score
            naive_mask_scores = np.zeros_like(intensities, dtype=float)
            naive_mask_scores[naive_mask] = 1.0

            # Connectivity score - strongly prefer voxels connected to centroid
            # First get high-scoring voxels based on other criteria
            initial_scores = 0.5 * distance_scores + 0.5 * intensity_scores
            high_score_mask = initial_scores >= np.percentile(initial_scores, 70)

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
                distance_weight * distance_scores
                + intensity_range_weight * intensity_scores
                + connectivity_weight * connectivity_scores
                + naive_mask_weight * naive_mask_scores
            )

            return final_scores

        naive_mask = naive_mask_data[coords[:, 0], coords[:, 1], coords[:, 2]] > 0

        if dynamic_centroid:
            valid_intensities = (
                (intensities >= intensity_range[0])
                & (intensities <= intensity_range[1])
                & naive_mask
            )

            if not np.any(valid_intensities):
                raise ValueError(
                    "No voxels found within the specified intensity range in naive mask"
                )
            # Need to revise this is taking place correctly...
            # Find initial centroid based on intensity-weighted center of naive mask region
            weighted_coords = coords[valid_intensities]
            weights = intensities[valid_intensities]
            centroid = np.average(weighted_coords, weights=weights, axis=0)
            print(f"Using dynamic centroid: {centroid} in voxel space")

        else:
            # Replace the current coordinate transformation code with:
            # Initial MNI coordinates for pituitary
            mni_coords_pituitary = np.array(
                [0, 2, -32, 1]
            )  # Adding 1 for homogeneous coordinates
            # Get voxel coordinates using the inverse affine transformation
            voxel_coords = np.dot(np.linalg.inv(mask_img.affine), mni_coords_pituitary)[
                :3
            ]
            centroid = np.round(voxel_coords).astype(int)
            print(f"Using hard-coded centroid: {centroid} in voxel space")

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
        scores = calculate_clustering_scores(
            coords, intensities, centroid, intensity_range, naive_mask, t1_data.shape
        )

        # Create initial mask based on scores
        selected_voxels = scores >= min_score_threshold

        # Use KMeans for selecting the most relevant voxels
        if np.sum(selected_voxels) > 0:
            clustering = KMeans(n_clusters=1, n_init=10, random_state=42).fit(
                coords[selected_voxels]
            )

            centroid = clustering.cluster_centers_[0]  # Updated centroid
            distances = np.linalg.norm(coords[selected_voxels] - centroid, axis=1)

            # Define a cutoff threshold (e.g., 90th percentile of distances to filter out outliers)
            cutoff_distance = np.percentile(distances, cluster_dist_threshold * 100)
            selected_voxels[selected_voxels] = distances <= cutoff_distance

        # Create final binary mask
        final_mask = np.zeros_like(naive_mask_data)
        final_mask[
            coords[selected_voxels, 0],
            coords[selected_voxels, 1],
            coords[selected_voxels, 2],
        ] = 1

        # Save updated mask if requested
        if update_mask:
            new_img = nib.Nifti1Image(final_mask, mask_img.affine, mask_img.header)
            nib.save(new_img, self.pituitary_mask)

        print(f"Dynamic pituitary mask created with {np.sum(selected_voxels)} voxels")

        # Create a scores visualization volume
        scores_volume = np.zeros_like(naive_mask_data)
        scores_volume[coords[:, 0], coords[:, 1], coords[:, 2]] = scores

        # Save scores as a NIfTI file for visualization
        scores_img = nib.Nifti1Image(scores_volume, mask_img.affine, mask_img.header)
        scores_path = os.path.join(self.output_dir, "pituitary_scores.nii.gz")
        nib.save(scores_img, scores_path)

        # Show the original visualization with an additional scores view
        show_mri_slices(
            [
                self.final_t1_mni,
                self.pituitary_mask,
                self.centroid_mask,
                scores_path,
            ],
            slice_index=self.slice_indices,
            titles=["T1w MNI with mask and scores"],
            overlay=True,
            colormaps=[
                "gray",  # T1 image in grayscale
                "hot",  # Binary mask in hot colors
                "hot",  # Centroid in hot colors
                "viridis",  # Scores in viridis colormap
            ],
        )

        return coords[selected_voxels], final_mask

    def __create_probabilistic_pituitary_mask(
        self,
        mni_coords: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
            (x_range[0], y_range[0], z_range[0]),
            (x_range[1], y_range[1], z_range[1]),
        ),
        dynamic_centroid: bool = False,
    ) -> np.ndarray:
        """
        Create a probabilistic pituitary mask using both dynamic masking and region growing methods.

        Parameters:
        mni_coords: Tuple of boundary coordinates in MNI space
        dynamic_centroid: Whether to use dynamic centroid calculation

        Returns:
        np.ndarray: Probabilistic mask where values represent confidence (0-1)
        """
        print("Creating probabilistic pituitary mask using combined methods...")

        # Get the dynamic mask scores
        _, dynamic_mask = self.__create_dynamic_pituitary_mask(
            update_mask=False, mni_coords=mni_coords, dynamic_centroid=dynamic_centroid
        )

        # Get the region growing mask
        t1_img = nib.load(self.final_t1_mni)
        t1_data = t1_img.get_fdata()

        # Get centroid in voxel space
        mni_coords_pituitary = np.array([0, 2, -32, 1])
        voxel_coords = np.dot(np.linalg.inv(t1_img.affine), mni_coords_pituitary)[:3]
        centroid = np.round(voxel_coords).astype(int)

        region_mask = self.__region_growing(t1_data, tuple(centroid))

        # Create probabilistic mask
        prob_mask = np.zeros_like(dynamic_mask, dtype=float)

        # Assign probabilities based on voting
        prob_mask[dynamic_mask > 0] += 0.5  # 50% confidence from dynamic method
        prob_mask[region_mask > 0] += 0.5  # 50% confidence from region growing

        # Save probabilistic mask
        prob_mask_img = nib.Nifti1Image(prob_mask, t1_img.affine)
        self.pituitary_mask = os.path.join(
            self.output_dir, "prob_" + PITUITARY_MASK_FILE
        )
        nib.save(prob_mask_img, self.pituitary_mask)

        # Visualize results
        show_mri_slices(
            [self.final_t1_mni, self.pituitary_mask],
            slice_index=self.slice_indices,
            titles=["T1w MNI with Probabilistic Mask"],
            overlay=True,
            colormaps=["gray", "viridis"],  # T1 image  # Probability map
        )

        print(f"Created probabilistic mask with values:")
        print(f"- Both methods agree: {np.sum(prob_mask > 0.9)} voxels")
        print(
            f"- Single method only: {np.sum((prob_mask > 0.4) & (prob_mask < 0.6))} voxels"
        )

        return prob_mask

    def __region_growing(self, image, seed, intensity_tol=150, max_voxels=1000):
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

        return mask

    def __alternative_pituitary_segmentation(self):
        """
        Alternative method for segmenting the pituitary gland using region growing.
        """
        if not self.final_t1_mni:
            raise ValueError(
                "Error: MRI not registered to MNI space. Run coregister_to_mni first."
            )

        # Load MRI data
        t1_img = nib.load(self.final_t1_mni)
        t1_data = t1_img.get_fdata()

        mask_img = nib.load(self.pituitary_mask)

        # Load previously determined pituitary centroid (in voxel space)
        mni_coords_pituitary = np.array(
            [0, 2, -32, 1]
        )  # Adding 1 for homogeneous coordinates
        # Get voxel coordinates using the inverse affine transformation
        voxel_coords = np.dot(np.linalg.inv(mask_img.affine), mni_coords_pituitary)[:3]
        centroid = np.round(voxel_coords).astype(int)
        print(f"Using hard-coded centroid: {centroid} in voxel space")

        # Perform region growing segmentation
        mask = self.__region_growing(t1_data, tuple(centroid))

        # Save mask as NIfTI file
        output_nifti = nib.Nifti1Image(mask.astype(np.uint8), t1_img.affine)
        self.pituitary_mask = os.path.join(self.output_dir, f"rg_{PITUITARY_MASK_FILE}")
        nib.save(output_nifti, self.pituitary_mask)

        # Show the segmented region
        show_mri_slices(
            [self.final_t1_mni, self.pituitary_mask],
            slice_index=self.slice_indices,
            titles=["T1w MNI with Segmented Pituitary"],
            overlay=True,
            colormaps=["gray", "hot"],
        )

        print("Alternative region-growing segmentation saved as", self.pituitary_mask)

    def __get_pituitary_statistics(self) -> dict:
        """
        Calculate statistics about the detected pituitary region

        Returns:
        dict: Dictionary containing various statistics about the pituitary region
        """
        print("Calculating pituitary statistics...")

        if not self.pituitary_mask or not os.path.exists(self.pituitary_mask):
            raise ValueError(
                "Pituitary mask not found. Run __create_naive_pituitary_mask first."
            )

        mask_img = nib.load(self.pituitary_mask)
        mask_data = mask_img.get_fdata()
        t1_img = nib.load(self.final_t1_mni)
        t1_data = t1_img.get_fdata()

        # Get coordinates of mask voxels
        mask_coords = np.array(np.where(mask_data > 0)).T

        if len(mask_coords) == 0:
            raise ValueError("No voxels found in pituitary mask")

        # Calculate center of mass in voxel space
        center_of_mass = np.mean(mask_coords, axis=0)

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

    def __create_output_directory(self):
        """
        Create output directory structure.
        """
        output_dir = os.path.join(
            PATH_TO_PROCESSED_DATA,
            f"{self.subject_id}_3T_Structural_proc",
            self.subject_id,
        )
        os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir

    def __perform_brain_extraction(self):
        """
        Performs brain extraction using FSL's BET tool and extracts the brain including the pituitary.
        """
        print("Extracting the brain using BET (FSL)...")

        # Step 1: Brain Extraction using BET (Removes non-brain tissues like the skull)
        t1w_brain = os.path.join(self.output_dir, T1_BRAIN_FILE)
        t2w_brain = os.path.join(self.output_dir, T2_BRAIN_FILE)

        subprocess.run(
            [
                "bet",
                self.unprocessed_t1,
                t1w_brain,
                "-R" if robust_brain_extraction else "",
                "-f",
                str(fractional_intensity_t1),
                "-g",
                str(gradient_t1),
            ],
            check=True,
        )
        subprocess.run(
            [
                "bet",
                self.unprocessed_t2,
                t2w_brain,
                "-R" if robust_brain_extraction else "",
                "-f",
                str(fractional_intensity_t2),
                "-g",
                str(gradient_t2),
            ],
            check=True,
        )

        # Show a slice of brain-extracted T1w image
        show_mri_slices(
            [t1w_brain, t2w_brain],
            titles=["T1w Extracted Brain MRI", "T2w Extracted Brain MRI"],
        )

    def __motion_correction_and_smoothing(self):
        """
        Perform motion correction and smoothing using FEAT.
        """
        print("Correcting for motion artifact & smoothing using FEAT...")

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

        # Show motion correctoin
        show_mri_slices(
            [self.motion_corrected_t1w, self.motion_corrected_t2w],
            titles=["T1w Motion Corrected MRI", "T2w Motion Corrected MRI"],
        )

        # Apply Gaussian smoothing
        self.smoothed_t1w = os.path.join(self.output_dir, T1_SMOOTH_FILE)
        self.smoothed_t2w = os.path.join(self.output_dir, T2_SMOOTH_FILE)

        subprocess.run(
            [
                "fslmaths",
                self.motion_corrected_t1w,
                "-s",
                str(smoothing_sigma),
                self.smoothed_t1w,
            ],
            check=True,
        )
        subprocess.run(
            [
                "fslmaths",
                self.motion_corrected_t2w,
                "-s",
                str(smoothing_sigma),
                self.smoothed_t2w,
            ],
            check=True,
        )

        show_mri_slices(
            [self.smoothed_t1w, self.smoothed_t2w],
            titles=["T1w Smoothed MRI", "T2w Smoothed MRI"],
        )

    def __registration_and_normalization(self):
        """
        Perform registration and normalization using FLIRT and FNIRT.
        """
        print("Completing registration & normalization using FLIRT...")

        # Paths
        self.registered_t1w = os.path.join(self.output_dir, T1_SMOOTH_REGISTERED_FILE)
        self.affine_matrix = os.path.join(self.output_dir, T1_TO_T2_MAT_FILE)
        self.normalized_t1w = os.path.join(self.output_dir, T1_MC_REGISTERED_FILE)

        # FLIRT (Affine Registration)
        subprocess.run(
            [
                "flirt",
                "-in",
                self.smoothed_t1w,
                "-ref",
                self.smoothed_t2w,
                "-out",
                self.registered_t1w,
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
                self.normalized_t1w,
                "-applyxfm",  # Use affine transformation matrix to apply the registration
                "-init",
                self.affine_matrix,
            ],
            check=True,
        )

        print("Affine Normalization Completed. T1 MRI in T2 Space.")

        # Display results
        show_mri_slices(
            [self.normalized_t1w, self.motion_corrected_t2w],
            titles=["T1w Affine Normalized MRI", "T2w Motion Corrected MRI"],
        )
