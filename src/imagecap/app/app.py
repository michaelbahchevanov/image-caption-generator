import gradio as gr
from imagecap.model import BLIP

def run():
    model = BLIP.load("Salesforce/blip-image-captioning-large", "Salesforce/blip-image-captioning-large")
    interface = gr.Interface(fn=lambda image: model.predict(image).caption, inputs="image", outputs="text")
    interface.launch()