from pathlib import Path

from sample_augment.data.state import StateBundle
from sample_augment.steps.step import step


class DummyOutput(StateBundle):
    important_state: int
    # TODO test relative paths across OS with pydantic
    maybe_a_file: Path


@step()
def dummy(state: StateBundle, random_seed: float):
    # error when replacing with DummyOutput (makes sense :)
    print("hello world!")
    print(random_seed)
    print(state)
