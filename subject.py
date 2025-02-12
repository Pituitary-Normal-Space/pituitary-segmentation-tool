"""
File that contains the Subject class. This is a class that represents a subject, their details, and stores their MRI data.
"""

# Standard libs
import os
import subprocess
from typing import Literal

# Non-standard libs

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
from const import PATH_TO_PROCESSED_DATA, MNI_TEMPLATE
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

    def __str__(self) -> str:
        return f"Subject ID: {self.subject_id}, Age: {self.age}, Sex: {self.sex}, Processing Complete: {self.processeing_complete}, UnProcessed T1 Path: {self.unprocessed_t1}, UnProcessed T2 Path: {self.unprocessed_t2}{', Processed T1 Path: {self.processed_t1}, Processed T2 Path: {self.processed_t2}' if self.processeing_complete else ''}"

    def preprocess_MRIs(self):
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

        print("Preprocessing completed successfully. Outputs stored in:", self.output_dir)

        self.processeing_complete = True

    def overlay_MRIs(self):
        """
        Function to overlay the T1 and T2 MRI data of the subject.
        This step is performed **after** preprocessing is complete.
        The overlay allows for a visual comparison between the two modalities.
        """

        # Construct file paths for the preprocessed images
        self.overlay_output = os.path.join(self.output_dir, "T1w_T2w_overlay.nii.gz")

        # Ensure the preprocessed images exist before overlaying
        if not os.path.exists(self.registered_t1w) or not os.path.exists(self.motion_corrected_t2w):
            print("Error: Preprocessed MRI files not found. Run preprocess_MRIs first.")
            return

        # Overlay T1w and T2w images using FSL’s fslmaths
        subprocess.run(
            ["fslmaths", self.registered_t1w, "-add", self.motion_corrected_t2w, self.overlay_output],
            check=True,
        )

        # Show a slice of the overlayed image
        show_mri_slices([self.overlay_output], titles=["T1w T2w Overlay MRI"])

        print(f"Overlay completed. Output saved at {self.overlay_output}.")

    def coregister_to_mni(self, mni_template_path=MNI_TEMPLATE, previously_coregistered=False):
        """
        Coregisters both the T1 and T2 images to the MNI template using affine and nonlinear warping.

        The transformation is computed on the smoothed T1 image, but the transformation is later 
        applied to the non-smoothed, motion-corrected T1 and T2 images to avoid excessive blurring.
        """
        # Paths to save transformations
        self.affine_matrix = os.path.join(self.output_dir, "t1_to_mni_affine.mat")
        self.warp_field = os.path.join(self.output_dir, "t1_to_mni_warp.nii.gz")

        # Output paths for final non-blurry normalized images
        self.final_t1_mni = os.path.join(self.output_dir, "T1w_motion_corrected_normalized_to_mni.nii.gz")
        self.final_t2_mni = os.path.join(self.output_dir, "T2w_motion_corrected_normalized_to_mni.nii.gz")

        if not previously_coregistered:
            print("Step 1: Computing transformation using smoothed T1 image...")

            # Compute Affine Transformation using FLIRT
            subprocess.run(
                [
                    "flirt",
                    "-in", self.smoothed_t1w,  # Smoothed T1 used for registration
                    "-ref", mni_template_path,
                    "-out", os.path.join(self.output_dir, "T1w_affine_to_mni.nii.gz"),
                    "-omat", self.affine_matrix,
                ],
                check=True
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
                    f"--iout={os.path.join(self.output_dir, 'T1w_smoothed_normalized_to_mni.nii.gz')}",
                    f"--splineorder={str(spline_order)}",
                    f"--numprec={hessian_precision}",
                    "--verbose",
                ],
                check=True
            )

            print("Nonlinear registration (FNIRT) to MNI completed.")

        # Step 3: Apply the Transformation to Non-blurry Motion-Corrected Images

        print("Step 3: Applying transformations to motion-corrected images...")

        # Apply affine + nonlinear warp to motion-corrected T1
        subprocess.run(
            [
                "applywarp",
                "-i", self.normalized_t1w,  # Non-smoothed motion-corrected T1
                "-r", mni_template_path,
                "-w", self.warp_field,
                "-o", self.final_t1_mni,
            ],
            check=True
        )

        print(f"T1 Motion Corrected Image Registered to MNI: {self.final_t1_mni}")

        # Apply affine + nonlinear warp to motion-corrected T2
        subprocess.run(
            [
                "applywarp",
                "-i", self.motion_corrected_t2w,  # Non-smoothed motion-corrected T2
                "-r", mni_template_path,
                "-w", self.warp_field,
                "-o", self.final_t2_mni,
            ],
            check=True
        )

        print(f"T2 Motion Corrected Image Registered to MNI: {self.final_t2_mni}")

        # Show results
        show_mri_slices([self.final_t1_mni, self.final_t2_mni, mni_template_path], 
                    titles=["T1w Motion Corrected Normalized to MNI", "T2w Motion Corrected Normalized to MNI", "MNI Template"])

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
        t1w_brain = os.path.join(self.output_dir, "T1w_brain.nii.gz")
        t2w_brain = os.path.join(self.output_dir, "T2w_brain.nii.gz")

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

        self.motion_corrected_t1w = os.path.join(self.output_dir, "T1w_mc.nii.gz")
        self.motion_corrected_t2w = os.path.join(self.output_dir, "T2w_mc.nii.gz")

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
        self.smoothed_t1w = os.path.join(self.output_dir, "T1w_smooth.nii.gz")
        self.smoothed_t2w = os.path.join(self.output_dir, "T2w_smooth.nii.gz")

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
        self.registered_t1w = os.path.join(self.output_dir, "T1w_registered.nii.gz")
        self.affine_matrix = os.path.join(self.output_dir, "t1w_to_t2w.mat")
        self.normalized_t1w = os.path.join(self.output_dir, "T1w_normalized.nii.gz")

        # FLIRT (Affine Registration)
        subprocess.run(
            [
                "flirt",
                "-in", self.smoothed_t1w,
                "-ref", self.smoothed_t2w,
                "-out", self.registered_t1w,
                "-omat", self.affine_matrix,
            ],
            check=True,
        )

        # Apply the affine transformation to the T1 image (output is already aligned to T2)
        subprocess.run(
            [
                "flirt",
                "-in", self.motion_corrected_t1w,
                "-ref", self.motion_corrected_t2w,
                "-out", self.normalized_t1w,
                "-applyxfm",  # Use affine transformation matrix to apply the registration
                "-init", self.affine_matrix,
            ],
            check=True,
        )

        print("Affine Normalization Completed.")

        # Display results
        show_mri_slices([self.normalized_t1w, self.motion_corrected_t2w], titles=["T1w Affine Normalized MRI", "T2w Motion Corrected MRI"])
