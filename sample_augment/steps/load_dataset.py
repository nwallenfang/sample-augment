from typing import Optional

from sample_augment.config import Config
from sample_augment.data.state import StateBundle
from sample_augment.data.state_store import StateStore
from sample_augment.steps.step import Step


class LoadDataset(Step):
    @classmethod
    def get_input_state_bundle(cls):
        return StateBundle

    @classmethod
    def run(cls, state: StateBundle, params: Config) -> StateBundle:
        pass

    @staticmethod
    def load_from(self, store: StateStore):
        pass

    @staticmethod
    def check_environment() -> Optional[str]:
        pass
