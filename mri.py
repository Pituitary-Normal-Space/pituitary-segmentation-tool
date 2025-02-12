# Third party imports
import nibabel as nib
import matplotlib.pyplot as plt


def show_mri_slices(mri_paths, slice_index=128, titles=None):
    """
    Display a slice of multiple MRI images side by side at a given index.

    :param mri_paths: List of MRI image paths (NIfTI format).
    :param slice_index: Index of the slice to display.
    :param titles: List of titles for each MRI image.
    """
    # Check that the number of titles matches the number of MRI images
    if titles is not None and len(titles) != len(mri_paths):
        raise ValueError(
            "The number of titles must match the number of MRI images provided."
        )

    # Load MRI data for each image in the list
    images_data = [nib.load(path).get_fdata() for path in mri_paths]

    # Create subplots with the number of columns equal to the number of MRIs
    num_images = len(mri_paths)
    fig, ax = plt.subplots(
        1, num_images, figsize=(6 * num_images, 6)
    )  # Adjust width based on number of images

    # If only one image, ax is not an array, so make sure to handle this case
    if num_images == 1:
        ax = [ax]

    # Display each MRI slice in the corresponding subplot
    for i, (data, mri_path) in enumerate(zip(images_data, mri_paths)):
        ax[i].imshow(data[slice_index, :, :], cmap="gray")
        ax[i].set_title(
            f"Slice {slice_index} of {mri_path.split('/')[-1]}"
            if titles is None
            else f"{titles[i]}: Slice {slice_index}"
        )
        ax[i].axis("off")

    # Show the side-by-side plot
    plt.tight_layout()
    plt.show()
