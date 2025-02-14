"""
File that contains the Subject class. This is a class that represents a subject, their details, and stores their MRI data.
"""

# Standard libs
import os
import subprocess
from typing import Literal, Tuple

# Non-standard libs
import numpy as np
import nibabel as nib

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

    def __str__(self) -> str:
        return f"Subject ID: {self.subject_id}, Age: {self.age}, Sex: {self.sex}, Processing Complete: {self.processeing_complete}, UnProcessed T1 Path: {self.unprocessed_t1}, UnProcessed T2 Path: {self.unprocessed_t2}{', Processed T1 Path: {self.processed_t1}, Processed T2 Path: {self.processed_t2}' if self.processeing_complete else ''}"

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
            "Preprocessing completed successfully. Outputs stored in:", self.output_dir
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
            print("Error: Preprocessed MRI files not found. Run preprocess_MRIs first.")
            return

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

        print(f"Overlay completed. Output saved at {self.overlay_output}.")

    def coregister_to_mni(
        self,
        mni_template_path: str = MNI_TEMPLATE,
        previously_coregistered: bool = False,
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

        if not previously_coregistered:
            print("Step 1: Computing transformation using smoothed T1 image...")

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

            print("Affine registration to MNI completed.")

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

            print("Nonlinear registration (FNIRT) to MNI completed.")

        # Step 3: Apply the Transformation to Non-blurry Motion-Corrected Images

        print("Step 3: Applying transformations to motion-corrected images...")

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

        print(f"T1 Motion Corrected Image Registered to MNI: {self.final_t1_mni}")

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

        print(f"T2 Motion Corrected Image Registered to MNI: {self.final_t2_mni}")

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

    def set_up_pituitary_analysis(self) -> None:
        """
        This function assumes that the MRI data has been preprocessed and registered to MNI space.

        It sets up the necessary files and directories for pituitary analysis.

        :return: None
        """
        self.__create_output_directory()
        self.processeing_complete = True

    def locate_pituitary(
        self,
        mni_coords: Tuple[Tuple[int, int, int], Tuple[int, int, int]] = (
            (-10, -3, -26),
            (12, 10, -38),
        ),  # These coordinates were determined by me
    ) -> Tuple[float, float, float]:
        """
        Locate the pituitary gland within the processed T1-weighted MRI scan using FSL tools.

        :param mni_coords: A tuple containing two MNI coordinate bounds to define the search region.
        :return: The MNI coordinates of the detected pituitary gland location.
        """
        if not self.final_t1_mni:
            print(
                "Error: MRI not registered to MNI space. Run coregister_to_mni first."
            )
            return None

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

        print(f"Voxel Start: {voxel_coords}")
        print(f"Voxel Size: {voxel_size}")

        # Set start x, y, z and size x, y, z
        start_x, start_y, start_z = voxel_coords
        size_x, size_y, size_z = voxel_size

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
        slice_indices = {
            "Axial": voxel_coords[2] + (voxel_size[2] // 2),  # Z-slice
            "Sagittal": voxel_coords[0] + (voxel_size[0] // 2),  # X-slice
            "Coronal": voxel_coords[1] + (voxel_size[1] // 2),  # Y-slice
        }

        # Load the extracted pituitary mask for intensity thresholding
        pituitary_data = nib.load(self.pituitary_mask).get_fdata()
        print(
            f"Pituitary mask intensity range: {np.min(pituitary_data)} - {np.max(pituitary_data)}"
        )
        print(np.unique(pituitary_data))  # Check if there are any non-zero values
        min_intensity = np.min(
            pituitary_data[pituitary_data > 300]
        )  # Ignore background, background tissue, and vessels for now
        max_intensity = np.max(pituitary_data[pituitary_data < 800])
        highlight_threshold = (min_intensity, max_intensity)

        print(f"Highlighting intensities between {highlight_threshold}")

        # Visualize the detected region
        show_mri_slices(
            [self.pituitary_mask],
            slice_index=slice_indices,  # Pass all three slice indices
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
        print("Step 1: Brain extraction using BET (FSL)...")

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
        print("Step 2: Motion Correction & Smoothing using FEAT...")

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
        print("Step 3: Registration & Normalization using FLIRT...")

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

        print("Affine Normalization Completed.")

        # Display results
        show_mri_slices(
            [self.normalized_t1w, self.motion_corrected_t2w],
            titles=["T1w Affine Normalized MRI", "T2w Motion Corrected MRI"],
        )
