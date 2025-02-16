"""
Non-interactive MRI visualization functions.
"""

# Standard library imports
from typing import List, Tuple, Dict, Optional

# Non-standard library imports
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

# Internal imports
from config import show_images


def show_mri_slices(
    mri_paths: List[str],
    slice_index: Dict[str, int] = None,
    titles: List[str] = None,
    highlight_threshold: Tuple[int, int] = None,
    overlay: bool = False,
    colormaps: List[str] = None,  # Add colormaps parameter
) -> None:
    """
    Display axial, sagittal, and coronal slices of multiple MRI images side by side or overlaid.

    :param mri_paths: List of MRI image paths (NIfTI format).
    :param slice_index: Dictionary with slice indices for Axial (Z), Sagittal (X), and Coronal (Y).
    :param titles: List of titles for each MRI image.
    :param highlight_threshold: Tuple (low, high) to highlight a range of intensities.
    :param overlay: If True, overlay multiple images on top of each other instead of showing them separately.
    :param colormaps: List of colormaps to use for each image. Defaults to ['gray'] for single images or ['gray', 'hot', ...] for overlays.

    :return: None
    """
    if not show_images:
        return
    if titles is not None and len(titles) != len(mri_paths) and not overlay:
        raise ValueError(
            "The number of titles must match the number of MRI images provided."
        )
    elif overlay and len(mri_paths) < 2:
        raise ValueError("At least two images are required for overlay mode.")
    elif overlay and titles is not None and len(titles) != 1:
        raise ValueError("Only one title is needed for overlay mode.")

    images_data = [nib.load(path).get_fdata() for path in mri_paths]
    num_images = len(mri_paths)

    if overlay and num_images < 2:
        raise ValueError("At least two images are required for overlay mode.")

    if overlay and colormaps is not None and len(colormaps) != num_images:
        raise ValueError(
            "The number of colormaps must match the number of MRI images provided."
        )

    # Set default colormaps if none provided
    if colormaps is None:
        if overlay:
            colormaps = ["gray"] + ["hot"] * (
                len(mri_paths) - 1
            )  # First image gray, others hot
        else:
            colormaps = ["gray"] * len(mri_paths)  # All gray by default

    # Calculate number of rows needed
    num_rows = 1 if overlay else num_images
    fig, ax = plt.subplots(num_rows, 3, figsize=(12, 4 * num_rows))

    if num_rows == 1:
        ax = [ax]

    for i in range(num_rows):
        slices_list = []
        for data in images_data:
            if slice_index:
                slices = [
                    data[:, :, slice_index["Axial"]],  # Axial
                    data[slice_index["Sagittal"], :, :],  # Sagittal
                    data[:, slice_index["Coronal"], :],  # Coronal
                ]
            else:
                slices = [
                    data[:, :, data.shape[2] // 2],  # Axial
                    data[data.shape[0] // 2, :, :],  # Sagittal
                    data[:, data.shape[1] // 2, :],  # Coronal
                ]
            slices_list.append(slices)

        for j in range(3):  # For each view (Axial, Sagittal, Coronal)
            if overlay:
                # Show base image
                base_slice = np.flip(slices_list[0][j], axis=1)
                ax[i][j].imshow(base_slice.T, cmap=colormaps[0], origin="upper")

                # Overlay additional images with specified colormaps and transparency
                for idx in range(1, len(slices_list)):
                    overlay_slice = np.flip(slices_list[idx][j], axis=1)
                    # Normalize overlay data to [0, 1]
                    overlay_norm = (overlay_slice - overlay_slice.min()) / (
                        overlay_slice.max() - overlay_slice.min() + 1e-10
                    )
                    ax[i][j].imshow(
                        overlay_norm.T,
                        cmap=colormaps[idx],
                        alpha=1.0 / num_images,
                        origin="upper",
                    )
            else:
                # Original single image display
                slice_data_flipped = np.flip(slices_list[i][j], axis=1)
                ax[i][j].imshow(slice_data_flipped.T, cmap=colormaps[i], origin="upper")

                if highlight_threshold:
                    mask = (slice_data_flipped >= highlight_threshold[0]) & (
                        slice_data_flipped <= highlight_threshold[1]
                    )
                    ax[i][j].imshow(mask.T, cmap="Reds", alpha=0.3)

            ax[i][j].axis("off")

            if titles:
                if overlay:
                    ax[i][j].set_title(
                        f"{titles[i]} - {['Axial', 'Sagittal', 'Coronal'][j]}"
                    )
                else:
                    ax[i][j].set_title(
                        f"{titles[i]} - {['Axial', 'Sagittal', 'Coronal'][j]}"
                    )

    plt.tight_layout()
    plt.show()
