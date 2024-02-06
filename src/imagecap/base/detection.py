"""Pydantic model for model results."""
import numpy as np
from pydantic import BaseModel, ConfigDict
from torch import Tensor


class Detection(BaseModel):
    """Represents a detection result.

    Attributes
    ----------
        orig_img (np.ndarray | Tensor | None): The original image.
        boxes (np.ndarray | Tensor | None): The bounding boxes of the detected objects if they exist.
        masks (Tensor | None): The segmentation masks of the detected objects if they exist.
        confidences (np.ndarray | Tensor | None): The confidence scores of the detected objects if they are calculated.
        labels (np.ndarray | list[str] | None): The labels of the detected objects if they exist.
        caption (str | list[str] | None): The caption describing the detected objects if it exists.

    """

    # TODO: Further validation should be added when more reqs are known
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    orig_img: np.ndarray | Tensor | None = None
    boxes: np.ndarray | Tensor | None = None
    masks: Tensor | None = None
    confidences: np.ndarray | Tensor | None = None
    labels: np.ndarray | list[str] | None = None
    caption: str | list[str] | None = None
