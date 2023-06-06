from typing import Optional

from sample_augment.params import Params
from sample_augment.data.state import InputState, OutputState
from sample_augment.steps.step import Step


class DownloadGC10(Step):
    required_parameters = []  # ?

    @classmethod
    def run(cls, state: InputState, params: Params) -> OutputState:
        # TODO
        return OutputState()

    @staticmethod
    def check_environment() -> Optional[str]:
        return Step._check_package('kaggle')
