# Pituitary Segmentation for Normal Space Generation

Repository containing code that completes segmentation of the pituitary based on supplied unprocessed T1 and T2 MRI images of the human brain in NIfTI (Neuroimaging Informatics Technology Initiative) format.

Currently we support working directly with data from the [100 Unrelated Subjects dataset from the HumanConnectome Project](https://db.humanconnectome.org/app/template/SubjectDashboard.vm?subjectGroupName=100%20Unrelated%20Subjects).

### How To Use

Takes a directory containing unprocessed MRIs in the following structure:

- **unprocessed/**
  - **subject_map.csv**: file that contains information about subjects that we have MRIs for
  - **<subject_id>\_3T_Structural_unproc/<subject_id>/unprocessed/**
    - **T1w_MPR1/<subject_id>\_3T_T1w_MPR1.nii.gz**: T1 MRI for subject
    - **T2w_SPC1/<subject_id>\_3T_T2w_SPC1.nii.jz**: T2 MRI for subject

Structure can be edited in const.py

Once you have set up this directory structure you can run the main file. It will:

- Create a list of subject objects for each subject in subject map
- Preprocess the MRIs for each
- Overlay the T1 and T2 images
- Coregister T1 and T2 to MNI space (path to this file is defined in const. Please update with your path.)
- Segment the pituitary gland using various methods

Additionally you can run a hyperparameter tuning function that will complete a grid-search over relevant parameters and return the best parameters to maximize the DICE criteria. Please note doing so requires the population of the **ground_truth_masks** directory. Must contain folders with the title as the subject_id and a mask containing nii.gz file within.

#### Setting Up Locally

Note: You must have [FSL installed](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index) on your local machine. Additionally you must access the relevant images to run this script from the Human Connectome Project.

- FSL cannot be run on Windows machines without WSL (Windows Subsystem for Linux). The link above has instructions on this, you can also contact **qdy4zt@virginia.edu** for help.
- Download our package manager poetry (if you have not downloaded it already)
  ```bash
  pip install poetry
  ```
- I have created the pyproject.toml files so you don't have to worry about any of that. Just do the below.
- Add configuration to have venv in project directory

  ```bash
  poetry config virtualenvs.in-project true
  ```

- Set up virtual environment using poetry

  ```bash
  poetry install --no-root
  ```

- Now try either of these two:

  - Now you should have a created venv that you can switch into with the following command
    ```bash
    poetry shell
    ```
  - Run the program with this command

    ```bash
    python main.py
    ```

  - If this doesn't work run
    ```bash
    poetry run python main.py
    ```

### What It Does

The steps it follows are outlined below by broad concept.

1. Preprocessed the MRIs by completing motion correction, smoothing, and affine transformation of T1 to T2 space

- Before preprocessing

  - <image showing original>

- After preprocessing
  - <image showing after preprocessing>

2. Overlay T1 and T2 for visualization purposes (optional)

- Example
  - <overlayed image>

3. Coregister to MNI

- Images moved to MNI space
  - <example of images in MNI space>

4. Pituitary segmentation: creates a ROI around centroid and shifts the centroid iteratively based on the final mask the below methods create.

- a. _Score-based segmentation_: gives each voxel in ROI a score based on distance, intensity, and connectivity with centroid.
  - Sample mask via score-based segmentation
  - <mask>
- b. _Region-growing segmentation_: starting with the centroid adds X number of voxels to the mask based on similarity with nearest voxels established as part of the mask.
  - Sample mask via region-growing segmentation
  - <mask>
- c. _Combine_ the above
  - Sample mask of combined methods
  - <mask>
- d. _Appendage removal_: utilizes various methods to smooth mask and remove appendages not believed to be the infundibulum. Can be done without.
  - Final mask example
  - <mask>
