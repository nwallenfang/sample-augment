from sample_augment.data.artifact import Artifact
from sample_augment.steps.step import step


class DummyState(Artifact):
    important_state: int


@step
def dummy(state: DummyState, random_seed: float):
    print("hello world!")
    print(f"important state: {state.important_state}")
    print(f"random_seed: {random_seed}")


@step
def dummy_producer() -> DummyState:
    return DummyState(
        important_state=11,
        overwrite=True
    )
