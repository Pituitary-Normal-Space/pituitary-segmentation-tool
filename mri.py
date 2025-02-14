"""
Non-interactive MRI visualization functions.
"""

# Standard library imports
from typing import List, Tuple, Dict

# Non-standard library imports
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt


def show_mri_slices(
    mri_paths: List[str],
    slice_index: Dict[str, int] = None,
    titles: List[str] = None,
    highlight_threshold: Tuple[int, int] = None,
):
    """
    Display axial, sagittal, and coronal slices of multiple MRI images side by side.

    :param mri_paths: List of MRI image paths (NIfTI format).
    :param slice_index: Dictionary with slice indices for Axial (Z), Sagittal (X), and Coronal (Y).
    :param titles: List of titles for each MRI image.
    :param highlight_threshold: Tuple (low, high) to highlight a range of intensities.
    """
    if titles is not None and len(titles) != len(mri_paths):
        raise ValueError(
            "The number of titles must match the number of MRI images provided."
        )

    images_data = [nib.load(path).get_fdata() for path in mri_paths]
    num_images = len(mri_paths)
    fig, ax = plt.subplots(num_images, 3, figsize=(12, 4 * num_images))

    if num_images == 1:
        ax = [ax]

    for i, data in enumerate(images_data):
        slices = (
            [
                data[:, :, slice_index["Axial"]],  # Axial
                data[slice_index["Sagittal"], :, :],  # Sagittal
                data[:, slice_index["Coronal"], :],  # Coronal
            ]
            if slice_index
            else [
                data[:, :, data.shape[2] // 2],  # Axial
                data[data.shape[0] // 2, :, :],  # Sagittal
                data[:, data.shape[1] // 2, :],  # Coronal
            ]
        )

        for j, slice_data in enumerate(slices):
            # Flip the image data along the y-axis (vertically)
            slice_data_flipped = np.flip(slice_data, axis=1)

            ax[i][j].imshow(
                slice_data_flipped.T, cmap="gray", origin="upper"
            )  # Use flipped data here
            ax[i][j].axis("off")

            if highlight_threshold:
                mask = (slice_data_flipped >= highlight_threshold[0]) & (
                    slice_data_flipped <= highlight_threshold[1]
                )
                ax[i][j].imshow(mask.T, cmap="Reds", alpha=0.3)

            if titles:
                ax[i][j].set_title(
                    f"{titles[i]} - {['Axial', 'Sagittal', 'Coronal'][j]}"
                )

    plt.tight_layout()
    plt.show()
