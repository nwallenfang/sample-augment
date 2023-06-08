from sample_augment.data.artifact import Artifact
from sample_augment.steps.step import step


class DummyState(Artifact):
    important_state: int


@step()
def dummy(state: DummyState, random_seed: float):
    # error when replacing with DummyOutput (makes sense :)
    print("hello world!")
    print(f"important state: {state.important_state}")
    print(f"random_seed: {random_seed}")


# TODO fix the factory thingy
@step()
def dummy_producer() -> DummyState:
    return DummyState(
        important_state=11
    )
