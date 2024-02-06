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
