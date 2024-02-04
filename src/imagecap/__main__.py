import subprocess
from pathlib import Path


def ensure_setup_models():
    if not Path("models").exists():
        try:
            subprocess.check_call(["make", "setup/models"])
        except subprocess.CalledProcessError:
            print('Failed to download models.')
            return
    print('Models are installed.')

ensure_setup_models()
