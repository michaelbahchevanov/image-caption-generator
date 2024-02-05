import subprocess
from pathlib import Path

import click
from imagecap.app import app as gradio_app


def ensure_setup_models():
    if not Path("models").exists():
        try:
            subprocess.check_call(["make", "setup/models"])
        except subprocess.CalledProcessError:
            raise SystemExit("Error: Unable to download models. Please check your internet connection and try again.")

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
def process():
    print("process")

@click.group()
def imagecap():
    pass

imagecap.add_command(app, name="app")
imagecap.add_command(cli, name="cli")

if __name__ == "__main__":
    imagecap()
