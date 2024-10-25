"""
File that contains the Subject class. This is a class that represents a subject, their details, and stores their MRI data.
"""

# Standard libs
from typing import Literal

# Non-standard libs

# Local libs


class Subject:
    """
    Subject class that represents a subject, their details, and stores their MRI data.

    :param subject_id: The ID of the subject.
    :param age: The age range of the subject.
    :param sex: The sex of the subject.
    :param t1_path: The path to the T1 MRI data.
    :param t2_path: The path to the T2 MRI data.
    :param: disease_status: The pituitary disease status of the subject. Default is None, valid options are adenoma and carcinoma.
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
        return f"Subject ID: {self.subject_id}, Age: {self.age}, Sex: {self.sex}, Disease Status: {self.disease_status}, Processing Complete: {self.processeing_complete}"

    def preprocess_MRIs(self):
        """
        Function to preprocess the MRI data of the subject.
        """
        pass

    def overlay_MRIs(self):
        """
        Function to overlay the T1 and T2 MRI data of the subject.
        """
        pass
