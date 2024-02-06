# Imagecap - generating caption from images

Requirements:

* Ubuntu 16.04 LTS, MacOS 14.2, WSL2
* Python 3.10
* Make

## Setup

Clone this repository:

```sh
git clone https://github.com/michaelbahchevanov/image-caption-generator.git
```

If you are using SSH:

```sh
git clone git@github.com:michaelbahchevanov/image-caption-generator.git
```

Then setup the virtual environment and activate it using:

```sh
make setup/env
source .venv/bin/activate
```

Now download the models and dependencies using:

```sh
make setup
```

If the poetry installation doesn't work run:

```sh
make install-no-poetry
```

You should have the */models* folder containing the config and checkpoints for the models.

## Usage

For more detailed information of the different options run:

```python
python src/imagecap --help
```

### Demo app

You can run the app using the following script:

```python
python src/imagecap app run-app
```

This should run a Gradio app on: <http://127.0.0.1:7860>

### CLI

You can also run the image taggers and the image description generators using the CLI. For more information of the usage run:

```python
python src/imagecap cli --help
```

#### Generate image caption

```python
python src/imagecap cli caption-image --input_image_path your/path/to/image --min_length 200 --max_length 250
```

#### Grounded image caption generation

```python
python src/imagecap cli grounded-tag-image --input_image_path your/path/to/image --prompt "frog. caption. wall. cat. dog."
```

#### Generate image tags

```python
python src/imagecap cli tag-image --input_image_path your/path/to/image
```

### For sample images use the images under `exploration/assets`

## Limitations

There are known limitations:

* If you are running the Gradio app and using the `tag-generator` or the `grounded-tag-generator` the application fails to draw multiple segments and doesn't return labels.
* The YOLOv8 model predicts multiple bboxes but they are not returned/drawn in the app, labelling works in the CLI
* The `caption-generator` works with images of all sizes despite losing information on images above 4096x4096 substantially. It also hallucinates when prompted for a larger length of input/output

## Improvements

* The models should be exposed through a web API - REST or RPC
* YOLOv8 should return multiple bbox predictions
* Error handling and further data model validation is needed
* Models should be fine-tuned on some domain-specific downstream task (with the exception of SAM)
* Extend the CLI to save outputs rather than just echo them
* Containerise the app to make it easily distributable
* Evaluate the models and performance; come up with metrics; evaluate on a dataset
* Expose more information coming from the models
* Allow for further guidance of the output via prompting
* Experiment with different models
* Make the outputs more customisable using templates for the object detection models
* Utilise environment for config rather than hard-coding names/paths
