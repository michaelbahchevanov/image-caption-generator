"""Model predictor for BLIP model."""
from imagecap.base.detection import Detection
from imagecap.base.predictor import BasePredictor
from transformers import BlipForConditionalGeneration, BlipProcessor


class BLIP(BasePredictor):
    """BLIP predictor class.

    Methods
    -------
        predict(image, **kwargs): Generates a caption for the given image.
        load(blip_model_ckpt, blip_processor_ckpt): Loads the BLIP model and processor from the given checkpoints.

    """

    def __init__(self, model, processor):
        self.model = model
        self.processor = processor

    def predict(self, image, **kwargs) -> Detection:
        """Generate a caption for the given image.

        Args:
        ----
            image: The input image.
            **kwargs: Additional keyword arguments for caption generation.

        Returns:
        -------
            Detection: An object containing the generated caption and the original image.

        """
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
        """Load the BLIP model and processor from the given checkpoints.

        Args:
        ----
            blip_model_ckpt: The checkpoint path for the BLIP model.
            blip_processor_ckpt: The checkpoint path for the BLIP processor.

        Returns:
        -------
            An instance of the BLIP predictor class.

        """
        processor = BlipProcessor.from_pretrained(blip_processor_ckpt)
        model = BlipForConditionalGeneration.from_pretrained(blip_model_ckpt)
        return cls(model, processor)

    def __repr__(self):
        return self.__class__.__name__
