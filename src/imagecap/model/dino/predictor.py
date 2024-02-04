from imagecap.base.predictor import BasePredictor
from imagecap.base.detection import Detection
from groundingdino.util.inference import Model

class Dino(BasePredictor):
    def __init__(self, model):
        self.model = model

    def predict(self, image) -> Detection:
        detections, phrases = self.model.predict_with_caption(
            image=image,
            caption="dog. cat.",
        )
        return Detection(
            boxes=detections.xyxy,
            confidences=detections.confidence,
            caption=phrases,
            orig_img=image
        )

    @classmethod
    def load(cls, model_path: str, **kwargs) -> "Dino":
        model = Model(
            model_config_path=kwargs.get("model_config_path", None), 
            model_checkpoint_path=model_path, 
            device="cpu"
        )
        return cls(model)
    

if __name__ == "__main__":
    import numpy as np
    from PIL import Image

    model = Dino.load(model_config_path="models/dino/config/GroundingDINO_SwinT_OGC.py", model_path="models/dino/weights/groundingdino_swint_ogc.pth")
    img = np.array(Image.open("exploration/assets/meow_and_woof.jpg"))[..., ::-1]
    preds = model.predict(img)
    print(preds.caption)