from pathlib import Path
from typing import Optional

from pydantic import FilePath

from sample_augment.config import Config
from sample_augment.data.state import StateBundle
from sample_augment.steps.step import Step
from sample_augment.steps.step_decorator import step
from sample_augment.utils.log import log


class DummyOutput(StateBundle):
    important_state: int
    # TODO test relative paths across OS with pydantic
    maybe_a_file: Path


class DummyStep(Step):
    @classmethod
    def get_input_state_bundle(cls):
        # doesn't need specific InputState
        return StateBundle

    @classmethod
    def run(cls, state: StateBundle, config: Config) -> DummyOutput:
        return DummyOutput(important_state=42, maybe_a_file=Path('relative/madlad.txt'))

    @staticmethod
    def check_environment() -> Optional[str]:
        return Step._check_package('kaggle')


@step
def dummy_step(state: StateBundle, ):
    pass