from pathlib import Path
import os

# project_dir = Path(__file__).resolve().parents[2]
# ROOT_DIR = os.path.abspath(project_dir)


def project_path(path: str, create=False) -> str:
    """ Resolve sub_path with project root path """
    # TODO does this fail when calling like this? python path/to/script.py
    # path = Path(path) commented this out because of calling a script in StyleGAN, not sure..
    project_dir = Path(__file__).resolve().parents[2]
    full_path = os.path.join(project_dir, path)

    if not os.path.exists(full_path) and create:
        os.mkdir(full_path)  # TODO could break if parent dir doesn't exist

    return full_path
