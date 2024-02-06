"""Model predictor for SAM model."""
import numpy as np
import torch
from imagecap.base.detection import Detection
from imagecap.base.predictor import BasePredictor
from PIL import Image
from segment_anything import SamPredictor, build_sam


class Sam(BasePredictor):
    """Sam class represents a predictor for segmentation mask predictions using the SAM model.

    Methods
    -------
        predict(image, **kwargs): Generates a caption for the given image.
        load(model_path): Loads the SAM model.
        show_mask(mask, image, random_color=True): Displays the mask overlay on the image.

    """

    def __init__(self, model_path: str):
        self.model = SamPredictor(build_sam(model_path).to(device="cpu"))

    def predict(self, image, **kwargs) -> Detection:
        """Generate a segmentation mask for the given image.

        Args:
        ----
            image: The input image.
            **kwargs: Additional keyword arguments.

        Returns:
        -------
            The detection object containing the generated caption and other information.

        """
        self.model.set_image(image)
        boxes = kwargs.get("boxes", None)
        transformed_boxes = self.model.transform.apply_boxes_torch(
            torch.Tensor(boxes), image.shape[:2]
        ).to("cpu")
        masks, _, _ = self.model.predict_torch(
            point_coords=None,
            point_labels=None,
            boxes=transformed_boxes,
            multimask_output=False,
        )
        return Detection(masks=masks, orig_img=image, boxes=transformed_boxes)

    @classmethod
    def load(cls, model_path: str) -> "Sam":
        """Load the SAM model.

        Args:
        ----
            model_path: The path to the SAM model.

        Returns:
        -------
            The loaded SAM model.

        """
        return cls(model_path)

    @staticmethod
    def show_mask(mask, image, random_color=True):
        """Display the mask overlay on the image.

        Args:
        ----
            mask: The mask to be displayed.
            image: The image on which the mask will be overlaid.
            random_color: Whether to use a random color for the mask overlay.

        Returns:
        -------
            The image with the mask overlay.

        """
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

        annotated_frame_pil = Image.fromarray(image).convert("RGBA")
        mask_image_pil = Image.fromarray(
            (mask_image.cpu().numpy() * 255).astype(np.uint8)
        ).convert("RGBA")

        return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

    def __repr__(self):
        return self.__class__.__name__
