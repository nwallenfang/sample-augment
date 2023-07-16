import logging
import os
import shutil
import sys
from pathlib import Path

from sample_augment.core import step, Artifact
from sample_augment.utils.log import log


class GC10Folder(Artifact):
    serialize_this = False
    image_dir: Path
    label_dir: Path


@step()
def download_gc10(shared_directory: Path) -> GC10Folder:
    import kaggle
    logging.debug(f"GC10 shared_dir: {shared_directory}")
    gc10_path = shared_directory / 'gc10'
    lable_dir = gc10_path / 'lable'
    new_lable_dir = shared_directory / 'gc10_labels'

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

        # rename class dirs to fix torch ImageFolder (else the class '10' comes between '1' and '2)
        for i in range(1, 11):
            old_dir = gc10_path / str(i)
            new_dir = gc10_path / str(i).zfill(2)
            os.rename(old_dir, new_dir)
    else:
        log.info("Skipping GC10 since dir exists.")

    assert gc10_path.is_dir()
    assert new_lable_dir.is_dir()
    return GC10Folder(image_dir=gc10_path, label_dir=new_lable_dir)

# TODO step: create gc10-mini
