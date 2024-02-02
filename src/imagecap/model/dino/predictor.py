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
    def from_local(cls, model_path: str):
        model = Model(
            model_config_path="models/dino/config/GroundingDINO_SwinT_OGC.py", 
            model_checkpoint_path="models/dino/weights/groundingdino_swint_ogc.pth", 
            device="cpu"
        )
        return cls(model)
    

if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    model = Dino.from_local("sure_bro")
    img = np.array(Image.open("exploration/assets/meow_and_woof.jpg"))[..., ::-1]
    preds = model.predict(img)
    print(preds.caption)