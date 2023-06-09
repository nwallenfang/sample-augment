from pathlib import Path
from typing import Optional

from sample_augment.core.artifact import Artifact
from sample_augment.core.step import Step
from sample_augment.core.config import Config
from sample_augment.utils.log import log


class GC10Folder(Artifact):
    imagefolder_path: Path
    gc10_label_dir: Path


class DownloadGC10(Step):
    @classmethod
    def get_input_state_bundle(cls):
        # doesn't need specific InputState
        return Artifact

    @classmethod
    def run(cls, state: Artifact, config: Config) -> GC10Folder:
        import kaggle
        import shutil

        log.info("Downloading GC10 dataset from kaggle..")
        kaggle.api.authenticate()  # throws IOError if API key is not configured
        kaggle.api.dataset_download_files('alex000kim/gc10det',
                                          path=config.raw_dataset_path,
                                          unzip=True)
        # move lable dir to
        lable_dir = config.raw_dataset_path / 'lable'
        new_lable_dir = config.root_directory / 'gc10_labels'

        shutil.move(lable_dir, new_lable_dir)

        return GC10Folder(imagefolder_path=config.raw_dataset_path, gc10_label_dir=new_lable_dir)

    @staticmethod
    def check_environment() -> Optional[str]:
        return Step._check_package('kaggle')


if __name__ == '__main__':
    # DownloadGC10().run(DownloadGC10.DownloadGC10InputState())
    # TODO the current method signature makes it pretty difficult to run a step individually
    #   I would prefer if there was a Config subset and maybe even a way of calling the method
    #   just by passing the state parameters individually
    pass
