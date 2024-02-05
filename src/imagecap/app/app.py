import gradio as gr
from imagecap.model import BLIP, YOLOv8, Dino, Sam
from enum import Enum

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
        return blip_model.predict(image).caption
    elif demo_type == DemoType.tag_generator.value:
        return yolo_model.predict(image).labels
    elif demo_type == DemoType.grounded_tag_generator.value:
        dino_preds = dino_model.predict(image, prompt=prompt)
        # sam_preds = sam_model.predict(image, boxes=dino_preds.boxes)
        return dino_preds.labels

def run():
    
    iface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Dropdown(choices=[demo_type.value for demo_type in DemoType]),
            gr.Image(),
            gr.Textbox(label="Prompt", placeholder="Type a prompt here...", visible=True)
        ],
        outputs="text"
    )
    iface.launch()
