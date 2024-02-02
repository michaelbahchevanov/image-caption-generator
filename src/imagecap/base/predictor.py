from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Type
import numpy as np
from imagecap.base.detection import Detection
import torch

class BasePredictor(ABC):

    @abstractmethod
    def predict(self: "BasePredictor", image: np.ndarray | torch.Tensor) -> Detection:
        ...

    @classmethod
    @abstractmethod
    def from_local(cls: Type["BasePredictor"], model_path: str) -> "BasePredictor":
        ...

    def __repr__(self):
        return self.__class__.__name__
    