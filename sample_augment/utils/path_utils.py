from pathlib import Path
import os

from sample_augment.utils import log


def project_path(path: str, create=False) -> str:
    """ Resolve sub_path with project root path """
    # TODO does this fail when calling like this? python path/to/script.py
    # path = Path(path) commented this out because of calling a script in StyleGAN, not sure..
    project_dir = Path(__file__).resolve().parents[2]
    full_path = os.path.join(project_dir, path)

    if not os.path.exists(full_path) and create:
        os.mkdir(full_path)  # TODO could break if parent dir doesn't exist

    return full_path


def _read_root_diretory() -> Path:  # read root_dir and experiment name from env file or system variables
    current_file_path = Path(__file__)
    project_root = current_file_path.parent.parent.parent
    env_path = project_root / '.env'
    assert env_path.is_file(), "expecting .env file in project root with ROOT_DIR variable"
    with open(env_path, 'r') as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            key, value = line.split("=", 1)

            assert key == 'ROOT_DIR'
            _root_directory = Path(value)
            if not _root_directory.is_dir():
                _root_directory = project_root / "data"
                assert _root_directory.is_dir()
                log.info(f"Using default root_directory {_root_directory}")

            break
    return _root_directory


root_directory: Path = _read_root_diretory()
