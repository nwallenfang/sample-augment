from typing import Optional

from sample_augment.prototype.step import Step
from sample_augment.prototype.params import Params
from sample_augment.prototype.state import InputState
from sample_augment.utils.log import log


class DownloadGC10(Step):
    required_parameters = []  # ?

    @classmethod
    def run(cls, state: InputState, params: Params):
        import kaggle

        log.info("Downloading GC10 dataset from kaggle..")
        kaggle.api.authenticate()  # throws IOError if API key is not configured
        kaggle.api.dataset_download_files('alex000kim/gc10det',
                                          path=params.raw_dataset_path,
                                          unzip=True)

    @staticmethod
    def check_environment() -> Optional[str]:
        return Step._check_package('kaggle')
