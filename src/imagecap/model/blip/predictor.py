from imagecap.base.detection import Detection
from imagecap.base.predictor import BasePredictor
from transformers import BlipForConditionalGeneration, BlipProcessor


class BLIP(BasePredictor):

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def predict(self, image, **kwargs) -> Detection:
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model.generate(
            **inputs,
            max_length=kwargs.get("max_length", 100),
            min_length=kwargs.get("min_length", 50),
            no_repeat_ngram_size=2
        )
        decoded_output = self.processor.decode(
            outputs[0], skip_special_tokens=True, truncate=True
        )
        return Detection(caption=decoded_output, orig_img=image)

    @classmethod
    def load(cls, blip_model_ckpt, blip_processor_ckpt) -> "BLIP":
        processor = BlipProcessor.from_pretrained(blip_processor_ckpt)
        model = BlipForConditionalGeneration.from_pretrained(blip_model_ckpt)
        return cls(model, processor)

    def __repr__(self):
        return self.__class__.__name__
