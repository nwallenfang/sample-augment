from pathlib import Path

from sample_augment.core import Experiment
from sample_augment.core.artifact import Artifact
from sample_augment.core.step import step
from sample_augment.core.config import read_config


class DummyState(Artifact):
    important_state: int


@step
def dummy(state: DummyState, random_seed: float):
    print(f"important state: {state.important_state}")
    print(f"random_seed: {random_seed}")


@step
def dummy_producer() -> DummyState:
    return DummyState(
        important_state=11
    )


def test_dummy_step():
    # TODO run dummy step with some simple interface
    #  like Experiment.run("Dummy", config=Config.test_default)
    #  either passing an InputState like this
    #  Experiment.run("Dummy",
    #                  config=Config.test_default,
    #                  input_artifact=DummyState(..)
    #  or else automatically resolve the producers that are needed to
    #  provide the input artifact.
    config = read_config(Path(r"C:\Users\Nils\Documents\Masterarbeit\sample-augment\config.json"))
    experiment = Experiment(config=config)
    experiment.run(target_name="Dummy")

