from pathlib import Path

from sample_augment.core import Experiment
from sample_augment.core.step import import_step_modules
from sample_augment.data.dataset import ImageFolderPath
from sample_augment.main import read_config


def test_pipeline_with_gc10_mini():
    # TODO pass initial state to experiment
    # maybe pass explicit pipeline as well
    # TODO see in dependency management that only the necessary steps get run
    # TODO get proper testing config, maybe use testing fixture for it
    config = read_config(Path("config.json"))

    import_step_modules(root_modules=['test', 'data'])

    # create Experiment instance
    experiment = Experiment(config)
    experiment.run("ImageFolderToTensors",
                   initial_artifacts=[ImageFolderPath(
                       image_folder_path=Path(
                           r"C:\Users\Nils\Documents\Masterarbeit\sample-augment\data\gc10-mini"))])
