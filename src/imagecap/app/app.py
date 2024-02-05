from enum import Enum

import gradio as gr
import numpy as np
from imagecap.model import BLIP, Dino, Sam, YOLOv8

blip_model = BLIP.load("Salesforce/blip-image-captioning-large", "Salesforce/blip-image-captioning-large")
sam_model = Sam.load("models/sam/weights/sam_vit_h_4b8939.pth")
yolo_model = YOLOv8.load("models/yolo/weights/yolov8n.pt")
dino_model = Dino.load(model_config_path="models/dino/config/GroundingDINO_SwinT_OGC.py", model_path="models/dino/weights/groundingdino_swint_ogc.pth")

class DemoType(Enum):
    caption_generator = "caption-generator"
    tag_generator = "tag-generator"
    grounded_tag_generator = "grounded-tag-generator"

def predict(demo_type, image, prompt):

    if demo_type == DemoType.caption_generator.value:
        return blip_model.predict(image, prompt=prompt).caption, None
    elif demo_type == DemoType.tag_generator.value:
        preds = yolo_model.predict(image)
        sam_preds = sam_model.predict(preds.orig_img, boxes=preds.boxes)
        annotated_frame_with_mask = np.copy(preds.orig_img)

        for i in range(len(sam_preds.masks)):
            annotated_frame_with_mask = sam_model.show_mask(sam_preds.masks[i][0], annotated_frame_with_mask)

        annotated_image = sam_model.show_mask(sam_preds.masks, sam_preds.orig_img)
        return set(preds.labels), annotated_image

    elif demo_type == DemoType.grounded_tag_generator.value:
        dino_preds = dino_model.predict(image, prompt=prompt)
        sam_preds = sam_model.predict(dino_preds.orig_img, boxes=dino_preds.boxes)
        annotated_frame_with_mask = np.copy(dino_preds.orig_img)

        for i in range(len(sam_preds.masks)):
            annotated_frame_with_mask = sam_model.show_mask(sam_preds.masks[i][0], annotated_frame_with_mask)

        return set(dino_preds.labels), annotated_frame_with_mask

def run():
    
    iface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Dropdown(choices=[demo_type.value for demo_type in DemoType], value=DemoType.caption_generator.value, label="Demo Type"),
            gr.Image(),
            gr.Textbox(label="Prompt", placeholder="Grounding prompt.", visible=True)
        ],
        outputs=["text", "image"]
    )
    iface.launch()
