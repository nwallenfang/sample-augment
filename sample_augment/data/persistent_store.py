import abc
import os.path
from abc import abstractmethod
from pathlib import Path

from pydantic import ValidationError

from sample_augment.config import Config
from sample_augment.data.artifact import ArtifactStore
from sample_augment.utils import log


class PersistentStore(abc.ABC):
    @abstractmethod
    def load_from_config(self, config: Config) -> ArtifactStore:
        # we'll have to see, maybe we'll only need the config_id (hash)
        pass

    @abstractmethod
    def save(self, artifacts: ArtifactStore, config: Config):
        pass


class DiskPersistentStore(PersistentStore):
    hash_prefix_length = 6
    root_dir: Path
    json_subdir: str = "states/"

    def __init__(self, root_dir: Path):
        self.root_dir = root_dir

    def save(self, artifacts: ArtifactStore, config: Config):
        config_hash = config.get_hash()[:self.hash_prefix_length]
        json_dir = self.root_dir / self.json_subdir
        if not os.path.isdir(json_dir):
            os.mkdir(json_dir)  # TODO error handling

        # TODO how to handle the state file already existing?
        with open(json_dir / f"state_{config_hash}.json", 'w') as json_file:
            json_file.write(artifacts.json(indent=4))
        with open(json_dir / f"config_{config_hash}.json", 'w') as json_file:
            json_file.write(config.json(indent=4))

    def load_from_config(self, config: Config) -> ArtifactStore:
        config_hash = config.get_hash()[:self.hash_prefix_length]
        json_dir = self.root_dir / self.json_subdir

        state_path = json_dir / f"state_{config_hash}.json"
        _config_path = json_dir / f"config_{config_hash}.json"

        # TODO verify that those configs are identical

        if state_path.exists():
            try:
                state = ArtifactStore.parse_file(state_path)
            except ValidationError as err:
                log.error(str(err))
                log.warning(f"Failed to parse {state_path.name}, starting with empty ArtifactStore.")
                return ArtifactStore()
            # TODO verify that State is valid, i.e. all file handles exist
            #  in theory this would
            log.info(f"Continuing with state {state_path.name}.")
            return state
        else:
            return ArtifactStore()
