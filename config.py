# Description: Configuration file for the project.
# Allows you to manipulate the parameters of the preprocessing pipeline and NL space development without changing the main code.

####################################################
# Pituitary masking parameters
####################################################
#
# Naive Masking Parameters
#
# This range of values is the X, Y, and Z coordinates of the pituitary gland in the MRI image.
# These are values of MNI space coordinates.
x_range = (-10, 12)
y_range = (-3, 10)
z_range = (-26, -39)
# This is the range of intensities that are considered to be part of the pituitary gland.
# This is used to create a mask of the pituitary gland.
intensity_range = (300, 800)


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
spline_order = 2  # 2 or 3, default is 3 but 2 is faster
# Its value can be either float or double (default) and it specifies the precision that the hessian H is calculated and stored in. Changing this to float will decrease the amount of RAM needed to store H and will hence allow one to go to slightly higher warp-resolution. The default is double since that is what we have used for most of the testing and validation.
hessian_precision = (
    "float"  # float or double, default is dobule but float is faster and takes less ram
)
