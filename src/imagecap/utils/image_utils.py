"""Module containing utility functions for image manipulation."""

_MAX_IMAGE_DIM = 1600


def ensure_image_dimensions(image) -> None:
    """Ensure that the input image dimensions are within the acceptable range.

    Args:
    ----
        image: The input image.

    Raises:
    ------
        ValueError: If the image dimensions are greater than the maximum allowed dimensions.

    """
    if image.shape[0] >= _MAX_IMAGE_DIM or image.shape[1] >= _MAX_IMAGE_DIM:
        raise ValueError("Image dimensions should be max 1600x1600")
