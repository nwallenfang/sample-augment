from typing import Optional

from sample_augment.data.state import InputState, OutputState
from sample_augment.params import Params

from sample_augment.steps.step import Step
from sample_augment.data.state_store import StateStore


class LoadDataset(Step):
    @classmethod
    def run(cls, state: InputState, params: Params) -> OutputState:
        pass

    @staticmethod
    def load_from(self, store: StateStore):
        pass

    @staticmethod
    def check_environment() -> Optional[str]:
        pass
