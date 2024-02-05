import numpy as np
from imagecap.model import BLIP, YOLOv8, Dino
from PIL import Image

blip_model = BLIP.load(
    "Salesforce/blip-image-captioning-large",
    "Salesforce/blip-image-captioning-large",
)

yolo_model = YOLOv8.load("models/yolo/weights/yolov8n.pt")

dino_model = Dino.load(
    model_config_path="models/dino/config/GroundingDINO_SwinT_OGC.py",
    model_path="models/dino/weights/groundingdino_swint_ogc.pth",
)


def get_image_caption(input_image_path, min_length, max_length):
    image = np.array(Image.open(input_image_path))
    caption = blip_model.predict(
        image, min_length=min_length, max_length=max_length
    ).caption
    return caption


def get_image_tags(input_image_path):
    image = np.array(Image.open(input_image_path))
    tags = yolo_model.predict(image).labels
    return tags


def get_grounded_image_tags(input_image_path, prompt):
    image = np.array(Image.open(input_image_path))
    tags = dino_model.predict(image, prompt=prompt).labels
    return tags
