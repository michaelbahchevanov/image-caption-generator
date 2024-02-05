import numpy as np
from pydantic import BaseModel, ConfigDict
from torch import Tensor


class Detection(BaseModel):
    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    orig_img: np.ndarray | Tensor | None = None
    boxes: np.ndarray | Tensor | None = None
    masks: Tensor | None = None
    confidences: np.ndarray | Tensor | None = None
    labels: np.ndarray | list[str] | None = None
    caption: str | list[str] | None = None
