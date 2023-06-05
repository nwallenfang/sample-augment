from typing import Optional

from prototype.experiment_step import ExperimentStep
from prototype.state import State, StepState
from utils.logging import logger


class DownloadGC10State(StepState):
    pass


class DownloadGC10(ExperimentStep):
    required_configs = ['']  # ?

    @classmethod
    def run(cls, state: State):
        import kaggle

        logger.info("GC10 dataset not found, downloading from kaggle..")
        kaggle.api.authenticate()  # throws IOError if API key is not configured
        kaggle.api.dataset_download_files('alex000kim/gc10det',
                                          path=state.config.raw_dataset_path,
                                          unzip=True)

    @staticmethod
    def check_environment() -> Optional[str]:
        return ExperimentStep._check_package('kaggle')
