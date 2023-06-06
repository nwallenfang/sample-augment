from typing import Optional

from sample_augment.steps.step import Step
from sample_augment.params import Params
from sample_augment.data.state import InputState
from sample_augment.utils.log import log


class DownloadGC10(Step):
    required_parameters = []  # ?

    class MyInputState(InputState):
        pass

    # this nee
    CustomInputState = MyInputState

    @classmethod
    def run(cls, state: CustomInputState, params: Params):
        import kaggle

        log.info("Downloading GC10 dataset from kaggle..")
        kaggle.api.authenticate()  # throws IOError if API key is not configured
        kaggle.api.dataset_download_files('alex000kim/gc10det',
                                          path=params.raw_dataset_path,
                                          unzip=True)

    @staticmethod
    def check_environment() -> Optional[str]:
        return Step._check_package('kaggle')
