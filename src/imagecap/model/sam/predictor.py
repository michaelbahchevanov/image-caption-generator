import numpy as np
from segment_anything import build_sam, SamPredictor

from imagecap.base.predictor import BasePredictor
from imagecap.base.detection import Detection

class Sam(BasePredictor):
    def __init__(self, model_path: str):
        self.model = SamPredictor(build_sam(model_path).to(device="cpu"))

    # TODO: change name of image for sam at least, detection is not a good name for this especially
    def predict(self, image: Detection | np.ndarray) -> Detection:
        self.model.set_image(image.orig_img if isinstance(image, Detection) else image)
        masks, _, _ = self.model.predict_torch(
            point_coords = None,
            point_labels = None,
            boxes = image.boxes if isinstance(image, Detection) else None,
            multimask_output = False,
        )
        return Detection(
            masks=masks,
        )

    @classmethod
    def from_local(cls, model_path: str):
        return cls(model_path)

    def __repr__(self):
        return self.__class__.__name__
    
if __name__ == "__main__":
    from PIL import Image
    import numpy as np
    from imagecap.model.yolov8.predictor import YOLOv8

    model = Sam.from_local("models/sam/weights/sam_vit_h_4b8939.pth")
    model_yolo = YOLOv8.from_local("models/yolo/weights/yolov8n.pt")
    img = np.array(Image.open("exploration/assets/meow_and_woof.jpg"))[..., ::-1]
    preds_yolo = model_yolo.predict(img)
    preds_sam = model.predict(preds_yolo)
    print(preds_sam.masks)

    