from pathlib import Path

import pytest

from sample_augment.core import Experiment
from sample_augment.core.step import find_steps
from sample_augment.data.dataset import ImageFolderPath
from sample_augment.main import read_config


@pytest.fixture
def experiment():
    config = read_config(Path("config.json"))
    find_steps(include=['test', 'data'])

    # create Experiment instance
    experiment_instance = Experiment(config)

    return experiment_instance


def test_pipeline_with_gc10_mini(experiment):
    # TODO pass initial state to experiment
    # maybe pass explicit pipeline as well
    # TODO see in dependency management that only the necessary steps get run
    # TODO get proper testing config, maybe use testing fixture for it
    experiment.run("ImageFolderToTensors",
                   additional_artifacts=[ImageFolderPath(
                       image_dir=Path(
                           r"C:\Users\Nils\Documents\Masterarbeit\sample-augment\data\gc10-mini"))])
