# Normal Space Generator

Repository containing code that generates a normal space for the pituitary based on supplied unprocessed T1 and T2 MRI images of the human brain in ... format.

Currently we support data from the [100 Unrelated Subjects dataset from the HCP](https://db.humanconnectome.org/app/template/SubjectDashboard.vm?subjectGroupName=100%20Unrelated%20Subjects).

### How To Use It

Mostly TBD... Takes a directory containing unprocessed MRIs in the following structure ... in the `data` folder and generates a normal space via these images. You will need to provide a csv that maps the subject IDs to age range and sex.

#### Setting Up Locally

Note: You must have [FSL installed](https://fsl.fmrib.ox.ac.uk/fsl/docs/#/install/index) on your local machine. Additionally you must access the relevant images to run this script from the Human Connectome Project.

- Also FSL cannot be run on Windows machines without WSL (Windows Subsystem for Linux). The link above has instructions on this, you can also contact me for help.
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

- Now you should have a created venv that you can switch into with the following command
  ```bash
  poetry shell
  ```
  - If this doesn't work run
    ```bash
    poetry run python main.py
    ```
- Run the program with this command
  ```bash
  python main.py
  ```

### What It Does

The steps it follows are outlined below by broad concept.

#### Processing

Details on processing here.

#### Many Other Things

To be continued
