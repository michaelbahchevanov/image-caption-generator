from imagecap.base.detection import Detection
from imagecap.base.predictor import BasePredictor
from ultralytics import YOLO


class YOLOv8(BasePredictor):

    def __init__(self, model: YOLO):
        self.model = model

    def predict(self, image) -> Detection:
        results = self.model.predict(image)
        return Detection(
            boxes=results[0].boxes.xyxy,
            confidences=results[0].boxes.conf,
            labels=[results[0].names[int(c)] for c in results[0].boxes.cls],
            orig_img=results[0].orig_img,
        )

    @classmethod
    def load(cls, model_path: str) -> "YOLOv8":
        return cls(YOLO(model=model_path))

    def __repr__(self):
        return self.__class__.__name__
