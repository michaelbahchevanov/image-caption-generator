from ultralytics import YOLO
from imagecap.base.predictor import BasePredictor
from imagecap.base.detection import Detection

class YOLOv8(BasePredictor):
    def __init__(self, model: YOLO):
        self.model = model

    def predict(self, image) -> Detection:
        results = self.model.predict(image)
        return Detection(
            boxes=results[0].boxes.xyxy,
            confidences=results[0].boxes.conf,
            labels=[results[0].names[int(c)] for c in results[0].boxes.cls],
            orig_img=results[0].orig_img
        )

    @classmethod
    def from_local(cls, model_path: str):
        return cls(YOLO(model=model_path))

    def __repr__(self):
        return self.__class__.__name__
    
if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    model = YOLOv8.from_local("models/yolo/weights/yolov8n.pt")
    img = np.array(Image.open("exploration/assets/meow_and_woof.jpg"))[..., ::-1]
    preds = model.predict(img)
    print(list(zip(preds.labels, preds.confidences.tolist())))