import abc
from abc import abstractmethod
from pathlib import Path

from sample_augment.params import Params
from sample_augment.data.state import State
from sample_augment.utils.paths import project_path


class StateStore(abc.ABC):
    @abstractmethod
    def load_from_config(self, config: Params) -> State:
        # we'll have to see, maybe we'll only need the config_id (hash)
        pass

    @abstractmethod
    def save(self, state: State):
        pass


class DiskStateStore(StateStore):
    root_dir: Path

    def __init__(self, root_dir: Path = project_path('data/store')):
        self.root_dir = root_dir

    def save(self, state: State):
        raise NotImplementedError()

    def load_from_config(self, config: Params) -> State:
        # for now, always return an empty state
        # TODO
        return State()