import os

# Paths for data from human connectome project
# Folders
PATH_TO_UNPROCESSED_DATA = "unprocessed"
PATH_TO_PROCESSED_DATA = "processed"
PATH_TO_GROUND_TRUTH_MASKS = "ground_truth_masks"
SUFFIX_SUBJECT_ID = "_3T_Structural_unproc"
SUFFIX_SUBJECT_ID_PROCESSED = "_3T_Structural_proc"
UNPROCESSED_DATA = os.path.join("unprocessed", "3T")
T1_DATA = "T1w_MPR1"
T2_DATA = "T2w_SPC1"
# MRI files
T1_FILE_NAME_SUFFIX = "_3T_T1w_MPR1.nii.gz"
T2_FILE_NAME_SUFFIX = "_3T_T2w_SPC1.nii.gz"
# Key map name for subject details
KEY_MAP_NAME = "subject_map.csv"

# Path to FSL resources
MNI_TEMPLATE = "/Users/charbelmarche/fsl/data/standard/MNI152_T1_0.5mm.nii.gz"  # Must replace with your local path


# Paths to store processed data

# Brain Extraction
# These files are the extracted brains from the T1 and T2 MRI images
T1_BRAIN_FILE = "T1w_brain.nii.gz"
T2_BRAIN_FILE = "T2w_brain.nii.gz"

# Smoothing
# Below are the smoothed T1 and T2 MRI images
T1_SMOOTH_FILE = "T1w_smooth.nii.gz"
T2_SMOOTH_FILE = "T2w_smooth.nii.gz"

# Motion Correction
# These files are the motion corrected T1 and T2 MRI images
T1_MC_FILE = "T1w_mc.nii.gz"
T2_MC_FILE = "T2w_mc.nii.gz"

# Registration of T1 to T2
# This is the matrix file for the transformation from T1 to T2 space
T1_TO_T2_MAT_FILE = "t1w_to_t2w.mat"
# This is the smoothed T1 MRI image registered to the T2 MRI image
T1_SMOOTH_REGISTERED_FILE = "T1w_smooth_registered_to_T2w.nii.gz"
# This is the MC not smoothed T1 MRI image registered to the T2 MRI image
T1_MC_REGISTERED_FILE = "T1w_registered_to_T2w.nii.gz"
# This is the warp file for the transformation from T1 to T2 space using FNIRT
T1_TO_T2_WARP_FILE = "t1_to_t2_warp.nii.gz"

# Overlaying images before MNI registration
# This is the overlay of the T1 and T2 MRI images
OVERLAY_FILE = "T1w_T2w_overlay.nii.gz"

# Registration of T1 to MNI and using results to register T2 to MNI
# This file is the affine transformation matrix from T1 to MNI space
T1_MNI_MAT_FILE = "t1_to_mni_affine.mat"
# This file is the affine transformed T1 MRI image to MNI space
T1_AFFINE_TO_MNI_FILE = "T1w_affine_to_mni.nii.gz"
# This file is the non-linearly transofrmed smooth T1 MRI image to MNI space
T1_SMOOTH_NONLIN_TO_MNI_FILE = "T1w_smoothed_normalized_to_mni.nii.gz"
# This is the warp file for the transformation from T1 to MNI space using FNIRT
T1_TO_MNI_WARP_FILE = "t1_to_mni_warp.nii.gz"
# These are the MC T1 and T2 MRI images in MNI space
T1_MNI_FILE = "T1w_motion_corrected_normalized_to_mni.nii.gz"
T2_MNI_FILE = "T2w_motion_corrected_normalized_to_mni.nii.gz"

# Pituitary Mask and Extraction
# This is the pituitary mask in MNI space
PITUITARY_MASK_FILE = "pituitary_mask.nii.gz"
# This is the centroid of the naive pituitary mask in MNI space
PITUITARY_CENTROID_FILE = "pituitary_centroid.nii.gz"
# Probabilistic mask prefix
PROB_MASK_PREFIX = "prob_"
# Regio growing mask prefix
REGION_GROWING_MASK_PREFIX = "region_growing_"
# Suffix for the final pituitary mask
PROB_PITUITARY_MASK = "pituitary_mask_scores.nii.gz"
