from typing import Optional

from sample_augment.config import Config
from sample_augment.data.artifact import Artifact
from sample_augment.data.persistent_store import PersistentStore
from sample_augment.steps.step import Step


class LoadDataset(Step):
    @classmethod
    def get_input_state_bundle(cls):
        return Artifact

    @classmethod
    def run(cls, state: Artifact, params: Config) -> Artifact:
        pass

    @staticmethod
    def load_from(self, store: PersistentStore):
        pass

    @staticmethod
    def check_environment() -> Optional[str]:
        pass
