from typing import Optional

from prototype.params import Params
from prototype.state import InputState, OutputState
from prototype.step import Step


class DownloadGC10(Step):
    required_parameters = []  # ?

    @classmethod
    def run(cls, state: InputState, params: Params) -> OutputState:
        # TODO
        return OutputState()

    @staticmethod
    def check_environment() -> Optional[str]:
        return Step._check_package('kaggle')
