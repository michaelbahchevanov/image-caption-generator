from groundingdino.util.inference import Model
from imagecap.base.detection import Detection
from imagecap.base.predictor import BasePredictor


class Dino(BasePredictor):

    def __init__(self, model):
        self.model = model

    def predict(self, image, **kwargs) -> Detection:
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
        model = Model(
            model_config_path=kwargs.get("model_config_path", None),
            model_checkpoint_path=model_path,
            device="cpu",
        )
        return cls(model)
