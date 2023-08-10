import os
from pathlib import Path
from sample_augment.utils import log


class DirectoryConfig:
    _root_dir = None  # Hidden class variable

    @staticmethod
    def _read_root_directory() -> Path:
        current_file_path = Path(__file__)
        project_root = current_file_path.parent.parent.parent

        # Highest priority method: reading from system env variable
        if 'ROOT_DIRECTORY' in os.environ:
            log.info(f"Root directory from environment variable: {os.environ['ROOT_DIRECTORY']}")
            return Path(os.environ['ROOT_DIRECTORY'])

        # Else there needs to be an env file containing the path
        env_path = project_root / '.env'
        if not env_path.is_file():
            log.warning(
                "Expected .env file in project root with ROOT_DIRECTORY variable. Please provide the file or set "
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
        return _root_directory

    @classmethod
    def get_root_dir(cls):
        if cls._root_dir is None:
            cls._root_dir = cls._read_root_directory()
        return cls._root_dir

    @classmethod
    def set_root_dir(cls, value: Path):
        cls._root_dir = value

    @classmethod
    def get_shared_dir(cls):
        return cls.get_root_dir() / "shared"


root_dir = DirectoryConfig.get_root_dir()
shared_dir = DirectoryConfig.get_shared_dir()


def set_root_directory(new_path: Path):
    DirectoryConfig.set_root_dir(new_path)
    global root_dir, shared_dir
    root_dir = new_path
    shared_dir = new_path / "shared"
