import numpy as np
from imagecap.base.detection import Detection
from imagecap.base.predictor import BasePredictor
from segment_anything import SamPredictor, build_sam


class Sam(BasePredictor):
    def __init__(self, model_path: str):
        self.model = SamPredictor(build_sam(model_path).to(device="cpu"))

    # TODO: change name of image for sam at least, detection is not a good name for this especially
    def predict(self, image, **kwargs) -> Detection:
        self.model.set_image(image)
        masks, _, _ = self.model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = kwargs.get("boxes", None),
            multimask_output = False,
        )
        return Detection(
            masks=masks,
        )

    @classmethod
    def load(cls, model_path: str) -> "Sam":
        return cls(model_path)

    def __repr__(self):
        return self.__class__.__name__
    
if __name__ == "__main__":
    import numpy as np
    from imagecap.model.yolov8.predictor import YOLOv8
    from PIL import Image

    model = Sam.load("models/sam/weights/sam_vit_h_4b8939.pth")
    model_yolo = YOLOv8.load("models/yolo/weights/yolov8n.pt")
    img = np.array(Image.open("exploration/assets/meow_and_woof.jpg"))[..., ::-1]
    preds_yolo = model_yolo.predict(img)
    preds_sam = model.predict(image=img, boxes=preds_yolo.boxes)
    print(preds_sam.masks)

    