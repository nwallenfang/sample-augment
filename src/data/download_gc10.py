from pathlib import Path
import os.path
import kaggle
import logging

from utils.paths import resolve_project_path

logging.getLogger().setLevel(logging.INFO)

def download_gc10(path):
    logging.info("GC10 dataset not found, downloading from kaggle..")
    kaggle.api.authenticate()  # throws IOError if API key is not configured
    kaggle.api.dataset_download_files('alex000kim/gc10det', path=path, unzip=True)

def load_gc10_if_missing():
    gc10_path = resolve_project_path("data/gc-10")

    if not os.path.exists(gc10_path):
        download_gc10(gc10_path)


if __name__ == '__main__':
    load_gc10_if_missing()
