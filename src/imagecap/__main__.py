"""Entry point for the imagecap application."""
import subprocess
from pathlib import Path

import click
from imagecap.app import app as gradio_app
from imagecap.cli import get_grounded_image_tags, get_image_caption, get_image_tags


def ensure_setup_models():
    """Ensure that the models are set up by checking if the 'models' directory exists.
    If the directory does not exist, it attempts to download the models using the 'make setup/models' command.
    If the download fails, it raises a SystemExit with an error message.
    """
    if not Path("models").exists():
        try:
            subprocess.check_call(["make", "setup/models"])
        except subprocess.CalledProcessError:
            raise SystemExit(
                "Error: Unable to download models. Please check your internet connection and try again."
            )


ensure_setup_models()


@click.group()
def app():
    """Click group for the Gradio app."""
    pass


@app.command()
def run_app():
    """Run the Gradio app as the main entry point function."""
    gradio_app.run()


@click.group()
def cli():
    """Click group for the command-line interface."""
    pass


@cli.command()
@click.option(
    "--input_image_path",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input image.",
)
@click.option("--min_length", type=click.INT, required=False, default=50)
@click.option("--max_length", type=click.INT, required=False, default=100)
def caption_image(input_image_path, min_length, max_length):
    """Generate a caption for the given input image.

    Args:
    ----
        input_image_path: The path to the input image file.
        min_length: The minimum length of the generated caption.
        max_length: The maximum length of the generated caption.

    """
    caption = get_image_caption(input_image_path, min_length, max_length)
    click.echo(caption)


@cli.command()
@click.option(
    "--input_image_path",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input image.",
)
def tag_image(input_image_path):
    """Tags an image using the specified input image path.

    Args:
    ----
        input_image_path: The path to the input image.

    """
    tags = get_image_tags(input_image_path)
    click.echo(tags)


@cli.command()
@click.option(
    "--input_image_path",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input image.",
)
@click.option("--prompt", "-p", type=click.STRING, required=True)
def grounded_tag_image(input_image_path, prompt):
    """Generate grounded image tags for the given input image.

    Args:
    ----
        input_image_path: The path to the input image file.
        prompt: The prompt to use for generating the tags.

    """
    tags = get_grounded_image_tags(input_image_path, prompt=prompt)
    click.echo(tags)


@click.group()
def imagecap():
    """Click group for the imagecap application."""
    pass


imagecap.add_command(app, name="app")
imagecap.add_command(cli, name="cli")

if __name__ == "__main__":
    imagecap()
