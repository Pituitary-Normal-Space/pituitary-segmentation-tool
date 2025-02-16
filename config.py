# Description: Configuration file for the project.
# Allows you to manipulate the parameters of the preprocessing pipeline and NL space development without changing the main code.

# If you want to see images for every step of the pipeline, set this to True.
# It will stop and wait for you to close the image to continue.
show_images = True

####################################################
# Pituitary masking parameters
####################################################
#
# Naive Masking Parameters
#
# This range of values is the X, Y, and Z coordinates of the pituitary gland in the MRI image.
# These are values of MNI space coordinates.
x_range = (-10, 12)
y_range = (-3, 9)
z_range = (-39, -27)
# This is the range of intensities that are considered to be part of the pituitary gland.
# This is used to create a mask of the pituitary gland.
intensity_range = (
    300,
    700,
)  # Will need to normalize intensity first and then play around with this.
#
# Dynamic Masking Parameters - Clustering-based
#
# This includes weights for various scores that are used to determine the pituitary gland.
# These scores are based on:
#   - Rroximity to the above intensity range
#   - The distance from naive centroid
#   - Connectivity with the centroid
#   - Being a part of the naive mask
# Must add up to 1.
distance_weight = 0.3
intensity_range_weight = 0.6
connectivity_weight = 0.1
naive_mask_weight = 0.0
# This is the minimum score threshold for a voxel to be a candidate for clustering as part of the pituitary gland.
min_score_threshold = 0.775  # Range 0-1
# This is the probability cut off to consider a voxel as part of the pituitary gland after clustering assigns probabilities to each voxel.
cluster_dist_threshold = 0.75  # Range 0-1
#
# Dynamic Masking Parameters - Region Growing-based
#
# This is the allowed intensity variation for region growing
intensity_tolerance = 150  # Variation in intensity allowed for region growing
# Maximum number of voxels to consider for region growing
max_voxels = 850

####################################################
# Preprocessing parameters
####################################################
#
####################################################
# Brain extraction parameters... not currently in use
#
# Fractional intensity threshold (0-1) for BET (0.5 is default).
fractional_intensity_t1 = 0.075
fractional_intensity_t2 = 0.075
# Gradient threshold for BET (higher values give larger brain outline estimates).
gradient_t1 = 0.25
gradient_t2 = 0.4
# Robust brain extraction for BET (reduce sensitivity to noise and intensity bias).
robust_brain_extraction = True
#
#
####################################################
# Smoothing parameters
#
# Sigma value for Gaussian smoothing filter (controls blurriness of the image).
# Small σ (e.g., 0.5 - 1.5) → Less smoothing, preserves fine details.
# Moderate σ (e.g., 2 - 3) → Balanced smoothing, reduces noise while keeping structure.
# Large σ (e.g., 5 - 10) → Heavy smoothing, blurs small details.
smoothing_sigma = 0.75
#
#
####################################################
# Non-linear registration parameters
#
# Specifies the order of the B-spline functions modelling the warp-fields. A spline-function is a piecewise continuous polynomial-function and the order of the spline determines the order of the polynomial and the support of the spline. In fnirt one can use splines of order 2 (quadratic) or 3 (the "well known" cubic B-spline).
spline_order = 3  # 2 or 3, default is 3 but 2 is faster
# Its value can be either float or double (default) and it specifies the precision that the hessian H is calculated and stored in. Changing this to float will decrease the amount of RAM needed to store H and will hence allow one to go to slightly higher warp-resolution. The default is double since that is what we have used for most of the testing and validation.
hessian_precision = "double"  # float or double, default is dobule but float is faster and takes less ram
