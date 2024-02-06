"""Abstract class for creating predictors."""
from abc import ABC, abstractmethod
from typing import Type

import numpy as np
import torch
from imagecap.base.detection import Detection


class BasePredictor(ABC):
    """Base class for predictors used in image caption generation."""

    @abstractmethod
    def predict(
        self: "BasePredictor", image: np.ndarray | torch.Tensor, **kwargs
    ) -> Detection:
        """Abstract method for predicting image captions.

        Args:
        ----
            image: The input image as a numpy array or torch tensor.
            **kwargs: Additional keyword arguments.

        Returns:
        -------
            The predicted image caption as a Detection object.

        """
        ...

    @classmethod
    @abstractmethod
    def load(cls: Type["BasePredictor"], model_path: str, **kwargs) -> "BasePredictor":
        """Abstract method for loading a predictor from a model path or named entry.

        Args:
        ----
            model_path: The path to the saved model or named entry.
            **kwargs: Additional keyword arguments.

        Returns:
        -------
            An instance of the loaded predictor.

        """
        ...

    def __repr__(self):
        return self.__class__.__name__
