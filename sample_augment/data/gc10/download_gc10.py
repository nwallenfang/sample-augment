import shutil
import sys
from pathlib import Path

from sample_augment.core import step, Artifact
from sample_augment.utils.log import log


class GC10Folder(Artifact):
    image_dir: Path
    label_dir: Path


@step()
def download_gc10(raw_data_directory: Path) -> GC10Folder:
    import kaggle
    gc10_path = raw_data_directory / 'gc10'
    lable_dir = gc10_path / 'lable'
    new_lable_dir = raw_data_directory / 'gc10_labels'

    if not gc10_path.is_dir():
        log.info("Downloading GC10 dataset from kaggle..")
        kaggle.api.authenticate()  # throws IOError if API key is not configured
        # noinspection PyBroadException
        try:
            kaggle.api.dataset_download_files('alex000kim/gc10det',
                                              path=gc10_path,
                                              unzip=True)
        except Exception as err:  # should catch kaggle's ApiExcpetion, which is not importable.
            log.error(str(err))
            log.error("kaggle APIError while trying to Download GC10. Make sure your access token is valid.")
            sys.exit(-1)

        # move lable (sic) dir outside the gc10 dir to make it easily readable by PyTorch ImageFolder
        # (which expects each class to have its own subdir)
        shutil.move(lable_dir, new_lable_dir)
    else:
        log.info("Skipping GC10 since dir exists.")

    assert gc10_path.is_dir()
    assert new_lable_dir.is_dir()
    return GC10Folder(image_dir=gc10_path, label_dir=new_lable_dir)

# TODO step: create gc10-mini
