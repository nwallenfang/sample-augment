from pathlib import Path
from sample_augment.utils import log


def _read_root_diretory() -> Path:  # read root_dir and experiment name from env file or system variables
    current_file_path = Path(__file__)
    project_root = current_file_path.parent.parent.parent

    # highest priority method: reading from system env variable
    import os
    if 'ROOT_DIRECTORY' in os.environ:
        log.info(f"Root directory from environment variable: {os.environ['ROOT_DIRECTORY']}")
        return Path(os.environ['ROOT_DIRECTORY'])

    # else there needs to be an env file containing the path
    env_path = project_root / '.env'
    if not env_path.is_file():
        log.warning("Expected .env file in project root with ROOT_DIRECTORY variable. Please provide the file or set "
                    "it manually with os.environ['ROOT_DIRECTORY']")

        assert env_path.is_file(), "Expected .env file in project root with ROOT_DIRECTORY variable. " \
                                   "Please provide the file or set it manually with os.environ['ROOT_DIRECTORY']"
    with open(env_path, 'r') as env_file:
        for line in env_file:
            line = line.strip()
            if not line or line.startswith('#'):
                continue

            key, value = line.split("=", 1)

            assert key.strip() == 'ROOT_DIRECTORY', f'key is {key}, was expecting ROOT_DIRECTORY'
            _root_directory = Path(value.strip())
            if not _root_directory.is_dir():
                _root_directory = project_root / "data"
                assert _root_directory.is_dir()
                log.info(f"Using default root_directory {_root_directory}")

            break
    log.debug(f"Root directory: {_root_directory}")
    return _root_directory


root_dir: Path = _read_root_diretory()
shared_dir: Path = root_dir / "shared"
