# Description: Configuration file for the project.
# Allows you to manipulate the parameters of the preprocessing pipeline and NL space development without changing the main code.


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
        #
        # Location and Intensity Parameters
        #
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
        # This is the range of intensities that are considered to be part of the pituitary gland.
        # This is used to create a mask of the pituitary gland.
        self.intensity_range = (
            300,
            800,
        )  # Will need to normalize intensity first and then play around with this.
        #
        # Maximum drift ROI/centroid can move in 1 direction before terminating the ideal centroid/ROI search.
        self.max_voxel_drift = 2  # Number of voxels
        #
        # Dynamic Masking Parameters - Score-based
        #
        # This includes weights for various scores that are used to determine the pituitary gland.
        # These scores are based on:
        #   - Rroximity to the above intensity range
        #   - The distance from naive centroid
        #   - Connectivity with the centroid
        # Must add up to 1.
        self.distance_weight = 0.3
        self.intensity_range_weight = 0.4
        self.connectivity_weight = 0.3
        # This is the number of high scoring neighbors (above min threshold when only considering distance and intensity) that a voxel must have to be considered connected to the centroid.
        self.high_quality_neighbors_to_consider_connected = 5
        # This is the minimum score threshold for a voxel to be a candidate for clustering as part of the pituitary gland.
        self.min_score_threshold = 0.8  # Range 0-1
        #
        # Dynamic Masking Parameters - Region Growing-based
        #
        # This is the allowed intensity variation for region growing
        self.intensity_tolerance = (
            225  # Variation in intensity allowed for region growing
        )
        # Maximum number of voxels to consider for region growing
        self.max_voxels = 8000
        #
        # Voting Parameters: How to incorporate region growing and score-based methods
        #
        # Weight for region growing method
        self.region_growing_weight = 0
        # Weight for score-based method
        self.score_based_weight = 1
        #
        # Boosting voxels with high scoring neighbors
        #
        # This is the number of high scoring neighbors (above min threshold when only considering distance and intensity) that a voxel must have to be considered connected to the centroid.
        self.num_neighbors_required_to_boost = 18  # Number between 0 and 26
        # This is the minimum score threshold for a voxel to be a candidate for boosting.
        self.min_score_to_boost_if_quality_neighbors = 0.5  # Range 0-1
        # This is the minimum score for a voxel to be considered a high scoring neighbor.
        self.min_score_considered_high_score = 0.75  # Range 0-1
        #
        # Appendage Removal Parameters
        #
        # Whether to do appendage removal or not
        self.do_appendage_removal = True
        # Voxels to be considered near the infundibulum area (0, 0, z_range's max)
        self.infundibulum_range = (
            3  # Range of voxels to consider near the infundibulum area
        )
        # Radius of the sphere to consider for appendage removal
        self.appendage_removal_radius = (
            1  # Radius of sphere to consider for appendage removal from 1 - ...
        )

        # Final Mask Selction Parameters
        # Threshold to consider a voxel as part of the pituitary gland based on final score
        self.final_score_threshold = 0.85  # Range 0-1

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
        self.gradient_t2 = 0.4
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


# Create a singleton instance
config = Config()
