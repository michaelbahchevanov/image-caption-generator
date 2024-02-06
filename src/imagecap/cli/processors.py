"""Functions to process images and generate captions and tags for the CLI."""
import numpy as np
from imagecap.model import BLIP, Dino, YOLOv8
from PIL import Image


def get_image_caption(input_image_path, min_length=50, max_length=100):
    """Generate a caption for an input image using the BLIP image captioning model.

    Args:
    ----
        input_image_path: The path to the input image file.
        min_length: The minimum length of the generated caption.
        max_length: The maximum length of the generated caption.

    Returns:
    -------
        The generated caption for the input image.

    """
    # TODO: Hardcoded model paths should be replaced with a config file
    blip_model = BLIP.load(
        blip_model_ckpt="Salesforce/blip-image-captioning-large",
        blip_processor_ckpt="Salesforce/blip-image-captioning-large",
    )
    image = np.array(Image.open(input_image_path))
    caption = blip_model.predict(
        image, min_length=min_length, max_length=max_length
    ).caption
    return caption


def get_image_tags(input_image_path):
    """Get the image tags using YOLOv8 model.

    Args:
    ----
        input_image_path: The path to the input image.

    Returns:
    -------
        The predicted tags for the image.

    """
    # TODO: Hardcoded model paths should be replaced with a config file
    yolo_model = YOLOv8.load(model_path="models/yolo/weights/yolov8n.pt")
    image = np.array(Image.open(input_image_path))
    tags = yolo_model.predict(image).labels
    return tags


def get_grounded_image_tags(input_image_path, prompt):
    """Get grounded image tags using the DINO model.

    Args:
    ----
        input_image_path: The path to the input image.
        prompt: The prompt for generating image tags.

    Returns:
    -------
        list: A list of grounded image tags.

    """
    # TODO: Hardcoded model paths should be replaced with a config file
    dino_model = Dino.load(
        model_config_path="models/dino/config/GroundingDINO_SwinT_OGC.py",
        model_path="models/dino/weights/groundingdino_swint_ogc.pth",
    )
    image = np.array(Image.open(input_image_path))
    tags = dino_model.predict(image, prompt=prompt).labels
    return tags
