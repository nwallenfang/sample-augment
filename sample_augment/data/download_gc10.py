import shutil
import sys
from pathlib import Path

from sample_augment.core import step, Artifact
from sample_augment.utils.log import log


class GC10Folder(Artifact):
    image_folder_path: Path
    gc10_label_dir: Path


@step(name="DownloadGC10")  # using custom name, since it's not quite camel case
def download_gc10(root_directory: Path) -> GC10Folder:
    import kaggle
    # TODO check if dataset is present
    gc10_path = root_directory / 'gc10'
    lable_dir = gc10_path / 'lable'
    new_lable_dir = root_directory / 'gc10_labels'

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
        log.debug("Skipping GC10 since dir exists.")

    assert gc10_path.is_dir()
    assert new_lable_dir.is_dir()
    return GC10Folder(image_folder_path=gc10_path, gc10_label_dir=new_lable_dir)
