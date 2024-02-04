from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Type
import numpy as np
from imagecap.base.detection import Detection
import torch

class BasePredictor(ABC):

    @abstractmethod
    def predict(self: "BasePredictor", image: np.ndarray | torch.Tensor, **kwargs) -> Detection:
        ...

    @classmethod
    @abstractmethod
    def load(cls: Type["BasePredictor"], model_path: str, **kwargs) -> "BasePredictor":
        ...

    def __repr__(self):
        return self.__class__.__name__
    