"""Model predictor for DINO model."""
from groundingdino.util.inference import Model
from imagecap.base.detection import Detection
from imagecap.base.predictor import BasePredictor


class Dino(BasePredictor):
    """Dino class for image caption generation using the Dino model.

    Methods
    -------
        predict(image, **kwargs): Generates a caption for the given image.
        load(model_path, **kwargs): Loads the Dino model from the specified path.

    """

    def __init__(self, model):
        self.model = model

    def predict(self, image, **kwargs) -> Detection:
        """Generate a caption for the given image.

        Args:
        ----
            image: The input image.
            **kwargs: Additional keyword arguments.

        Returns:
        -------
            Detection: The detection object containing the predicted caption and other information.

        """
        detections, phrases = self.model.predict_with_caption(
            image=image,
            caption=kwargs.get("prompt", ""),
        )
        return Detection(
            boxes=detections.xyxy,
            confidences=detections.confidence,
            caption=kwargs.get("prompt", ""),
            labels=phrases,
            orig_img=image,
        )

    @classmethod
    def load(cls, model_path: str, **kwargs) -> "Dino":
        """Load the Dino model from the specified path.

        Args:
        ----
            model_path: The path to the model checkpoint file.
            **kwargs: Additional keyword arguments.

        Returns:
        -------
            The loaded Dino model.

        """
        model = Model(
            model_config_path=kwargs.get("model_config_path", None),
            model_checkpoint_path=model_path,
            device="cpu",
        )
        return cls(model)
