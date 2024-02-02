import subprocess
from pathlib import Path

def ensure_downloads():
    if not Path("models").exists():
        print('Downloading models...')
        try:
            subprocess.check_call(["make", "download_models"])
        except subprocess.CalledProcessError:
            print('Failed to download models.')
            return
    print('Models are installed.')

ensure_downloads()