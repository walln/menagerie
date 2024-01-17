"""Utilities for image classification tasks."""
from albumentations import CenterCrop, Compose, Normalize, PadIfNeeded


def create_standard_image_transforms(height=224, width=224):
    """Create a standard image transformation pipeline.

    Images are guaranteed to be 224x224. Smaller images will be padded and larger images cropped.
    Images are also then normalized and converted to torch Tensors.
    """
    return Compose(
        [
            PadIfNeeded(min_height=height, min_width=width),
            CenterCrop(height=height, width=width),
            Normalize(),
        ]
    )
