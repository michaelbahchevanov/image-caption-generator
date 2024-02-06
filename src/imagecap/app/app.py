"""Gradio app for image captioning."""
from enum import Enum

import gradio as gr
import numpy as np
from imagecap.model import BLIP, Dino, Sam, YOLOv8

# TODO: Hardcoded model paths should be replaced with environment variables or constants.
blip_model = BLIP.load(
    "Salesforce/blip-image-captioning-large", "Salesforce/blip-image-captioning-large"
)
sam_model = Sam.load("models/sam/weights/sam_vit_h_4b8939.pth")
yolo_model = YOLOv8.load("models/yolo/weights/yolov8n.pt")
dino_model = Dino.load(
    model_config_path="models/dino/config/GroundingDINO_SwinT_OGC.py",
    model_path="models/dino/weights/groundingdino_swint_ogc.pth",
)


_MAX_IMAGE_DIM = 1600


def _ensure_image_dimensions(image) -> None:
    """Ensure that the input image dimensions are within the acceptable range.

    Args:
    ----
        image: The input image.

    Raises:
    ------
        ValueError: If the image dimensions are greater than the maximum allowed dimensions.

    """
    if image.shape[0] >= _MAX_IMAGE_DIM or image.shape[1] >= _MAX_IMAGE_DIM:
        raise ValueError("Image dimensions should be max 1600x1600")


class DemoType(Enum):
    """Enum representing the different available modes."""

    caption_generator = "caption-generator"
    tag_generator = "tag-generator"
    grounded_tag_generator = "grounded-tag-generator"


def predict_caption_generator(image, min_length, max_length):
    """Predicts a caption for the given image using the caption generator model (BLIP).

    Args:
    ----
        image: The input image for which to generate a caption.
        min_length: The minimum length of the generated caption.
        max_length: The maximum length of the generated caption.

    Returns:
    -------
        The predicted caption for the given image.

    """
    return blip_model.predict(
        image, min_length=min_length, max_length=max_length
    ).caption


def predict_tag_generator(image):
    """Generate tags for the given image using YOLOv8 for object detection and SAM for segmentation.

    Args:
    ----
        image: The input image.

    Returns:
    -------
        A tuple containing a set of predicted tags and an annotated image.

    """
    try:
        _ensure_image_dimensions(image)
        preds = yolo_model.predict(image)
        sam_preds = sam_model.predict(preds.orig_img, boxes=preds.boxes)
        annotated_frame_with_mask = np.copy(preds.orig_img)

        for i in range(len(sam_preds.masks)):
            annotated_frame_with_mask = sam_model.show_mask(
                sam_preds.masks[i][0], annotated_frame_with_mask
            )

        annotated_image = sam_model.show_mask(sam_preds.masks, sam_preds.orig_img)
        return set(preds.labels), annotated_image
    except ValueError as e:
        return str(e), None


def predict_grounded_tag_generator(image, prompt):
    """Generate grounded tags for an image based on a prompt.

    Args:
    ----
        image: The input image.
        prompt: The prompt for generating grounded tags.

    Returns:
    -------
        A tuple containing a set of grounded tags and the annotated image with masks.

    """
    try:
        _ensure_image_dimensions(image)
        dino_preds = dino_model.predict(image, prompt=prompt)
        sam_preds = sam_model.predict(dino_preds.orig_img, boxes=dino_preds.boxes)
        annotated_frame_with_mask = np.copy(dino_preds.orig_img)

        for i in range(len(sam_preds.masks)):
            annotated_frame_with_mask = sam_model.show_mask(
                sam_preds.masks[i][0], annotated_frame_with_mask
            )

        return set(dino_preds.labels), annotated_frame_with_mask
    except ValueError as e:
        return str(e), None


def run():
    """Run the Gradio app as the main entry point function."""
    with gr.Blocks() as demo:
        with gr.Tab(DemoType.caption_generator.value):
            with gr.Row():
                image_input_cap_gen = gr.Image()
                text_output_cap_gen = gr.Textbox(placeholder="Output image caption.")
            with gr.Row():
                min_val_cap_gen = gr.Slider(
                    label="Min Output Length", minimum=1, maximum=1000, value=50
                )
                max_val_cap_gen = gr.Slider(
                    label="Max Output Length", minimum=1, maximum=1000, value=100
                )
            btn_cap_gen = gr.Button("Generate Caption")

        with gr.Tab(DemoType.tag_generator.value):
            with gr.Row():
                image_input_tag_gen = gr.Image()
            with gr.Row():
                text_output_tag_gen = gr.Textbox(label="Output image tags.")
                image_output_tag_gen = gr.Image()
            btn_tag_gen = gr.Button("Generate Tags")

        with gr.Tab(DemoType.grounded_tag_generator.value):
            with gr.Row():
                image_input_grounded_gen = gr.Image()
                text_input_grounded_gen = gr.Textbox(
                    placeholder="Grounding prompt in the form: frog. tree. backpack. etc."
                )
            with gr.Row():
                image_output_grounded_gen = gr.Image()
                text_output_grounded_gen = gr.Textbox(label="Output grounded tags.")
            btn_grounded_gen = gr.Button("Generate Grounded Tags")

        btn_cap_gen.click(
            predict_caption_generator,
            inputs=[image_input_cap_gen, min_val_cap_gen, max_val_cap_gen],
            outputs=text_output_cap_gen,
        )
        btn_tag_gen.click(
            predict_tag_generator,
            inputs=image_input_tag_gen,
            outputs=[text_output_tag_gen, image_output_tag_gen],
        )
        btn_grounded_gen.click(
            predict_grounded_tag_generator,
            inputs=[image_input_grounded_gen, text_input_grounded_gen],
            outputs=[text_output_grounded_gen, image_output_grounded_gen],
        )

    demo.launch()
