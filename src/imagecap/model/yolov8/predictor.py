"""Model predictor for YOLOv8."""
from imagecap.base.detection import Detection
from imagecap.base.predictor import BasePredictor
from ultralytics import YOLO


class YOLOv8(BasePredictor):
    """YOLOv8 class for object detection using pretrained YOLOv8 model.

    Methods
    -------
        predict(image): Performs object detection on the given image and returns the detection results.
        load(model_path): Loads the YOLOv8 model from the specified model path.

    """

    def __init__(self, model: YOLO):
        self.model = model

    def predict(self, image) -> Detection:
        """Perform object detection on the given image and return the detection results.

        Args:
        ----
            image: The input image for object detection.

        Returns:
        -------
            The detection results containing bounding boxes, confidences, labels, and original image.

        """
        results = self.model.predict(image)
        return Detection(
            boxes=results[0].boxes.xyxy,
            confidences=results[0].boxes.conf,
            labels=[results[0].names[int(c)] for c in results[0].boxes.cls],
            orig_img=results[0].orig_img,
        )

    @classmethod
    def load(cls, model_path: str) -> "YOLOv8":
        """Load the YOLOv8 model from the specified model path.

        Args:
        ----
            model_path: The path to the YOLOv8 model.

        Returns:
        -------
            The loaded YOLOv8 model.

        """
        return cls(YOLO(model=model_path))

    def __repr__(self):
        return self.__class__.__name__
