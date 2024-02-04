import subprocess
from imagecap.app import app as gradio_app
from pathlib import Path

import click


def ensure_setup_models():
    if not Path("models").exists():
        try:
            subprocess.check_call(["make", "setup/models"])
        except subprocess.CalledProcessError:
            print("Failed to download models.")
            return

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
