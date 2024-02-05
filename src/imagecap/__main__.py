import subprocess
from pathlib import Path

import click
from imagecap.app import app as gradio_app
from imagecap.cli import get_grounded_image_tags, get_image_caption, get_image_tags


def ensure_setup_models():
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
    pass


@app.command()
def run_app():
    gradio_app.run()


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--input_image_path",
    "-i",
    type=click.Path(exists=True),
    required=True,
    help="Path to the input image.",
)
@click.option("--min_length", type=click.INT, required=True)
@click.option("--max_length", type=click.INT, required=True)
def caption_image(input_image_path, min_length, max_length):
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
    tags = get_grounded_image_tags(input_image_path, prompt=prompt)
    click.echo(tags)


@click.group()
def imagecap():
    pass


imagecap.add_command(app, name="app")
imagecap.add_command(cli, name="cli")

if __name__ == "__main__":
    imagecap()
