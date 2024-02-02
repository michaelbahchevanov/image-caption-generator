# Image Caption Generator

This folder contains prototypes and their accompanying research for an image description generator. The goal of this project is to develop a solution that can generate descriptive captions for input images.

## Experiments

### 1. Label Box Detection

In this experiment, I have used object detection techniques to detect the bounding box and label of objects in an image.

### 2. Image Segmentation

Image segmentation will be used to identify the exact position of objects in an image. This will help us generate more precise and detailed descriptions.

### 3. Combination of Label Box Detection and Segmentation with Grounding Prompts

By combining the results of label box detection and image segmentation, along with grounding prompts, we aim to improve the accuracy and contextual understanding of the generated captions.

### 4. CNN for Feature Extraction into a Sequence-to-Sequence Model

In this experiment, we will use a Convolutional Neural Network (CNN) to extract features from the input image. These features will then be fed into a sequence-to-sequence model (encoder-decoder) to generate captions.

### 5. Visual Transformer into Another Decoder

In the final experiment, we will explore the use of a visual transformer model. The visual transformer will be used to encode the input image, and the encoded representation will be passed to another decoder to generate captions.

## Why These Experiments?

Each experiment focuses on a different aspect of image understanding and caption generation. By combining these techniques, we aim to improve the accuracy, contextual understanding, and level of detail in the generated captions. Through these experiments, we hope to develop a robust and effective image description generator.

Please refer to the individual experiment folders for more details on the implementation and results.
