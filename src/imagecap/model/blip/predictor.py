from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

from imagecap.base.predictor import BasePredictor
from imagecap.base.detection import Detection

class BLIP(BasePredictor):
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def predict(self, image) -> Detection:
        inputs = self.processor(images=image, text="an image of",return_tensors="pt", max_length=50, truncation=True)
        outputs = self.model.generate(**inputs)
        decoded_output = self.processor.decode(outputs[0], skip_special_tokens=True)
        return Detection(
            caption=decoded_output,
            orig_img=image
        )

    @classmethod
    def from_local(cls, blip_model_ckpt, blip_processor_ckpt):
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
        return cls(model, processor)

    def __repr__(self):
        return self.__class__.__name__

if __name__ == "__main__":
    from PIL import Image
    import numpy as np

    model = BLIP.from_local("Salesforce/blip-image-captioning-large", "Salesforce/blip-image-captioning-large")
    img = np.array(Image.open("exploration/assets/meow_and_woof.jpg"))[..., ::-1]
    preds = model.predict(img)
    print(preds.caption)