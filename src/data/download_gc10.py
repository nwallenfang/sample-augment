from pathlib import Path
import os.path
import kaggle
import logging

logging.getLogger().setLevel(logging.INFO)

def download_gc10(path):
    logging.info("GC10 dataset not found, downloading from kaggle..")
    kaggle.api.authenticate()  # throws IOError if API key is not configured
    kaggle.api.dataset_download_files('alex000kim/gc10det', path=path, unzip=True)

def load_gc10_if_missing():
    project_dir = Path(__file__).resolve().parents[2]
    gc10_path = os.path.join(project_dir, "data", "gc-10")

    if not os.path.exists(gc10_path):
        load_gc10(gc10_path)


if __name__ == '__main__':
    load_gc10_if_missing()
