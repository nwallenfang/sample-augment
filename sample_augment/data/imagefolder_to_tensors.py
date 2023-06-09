from __future__ import annotations
import pickle
import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from sample_augment.core.config import Config
from sample_augment.core.artifact import Artifact
from sample_augment.core.step import Step
from sample_augment.utils import log


class SamplingAugDataset(TensorDataset):
    """
        PyTorch TensorDataset with the extension that we're also saving the paths to the
        original images.
    """

    def __init__(self, name: str, image_tensor: Tensor, label_tensor: Tensor,
                 img_paths: list, root_dir: Path):
        """
        Args:
            img_paths (list[Path]): list going from Tensor index to relative path of the given image
            root_dir (Path): for resolving the relative paths from img_ids
        """
        super().__init__(image_tensor, label_tensor)
        self.name = name
        self.root_dir = root_dir
        self.img_paths: list = img_paths

    def get_img_id(self, index: int) -> str:
        """
            img_id is just the filename without the file ending and the redundant `img_` prefix
        """
        if isinstance(self.img_paths[index], Path):
            img_id = str(self.img_paths[index].name)
            img_id = img_id.split('.')[0][4:]
        else:
            img_id = self.img_paths[index]
        return str(img_id)

    def get_img_path(self, index: int) -> Path:
        return Path.joinpath(self.root_dir, self.img_paths[index])

    @classmethod
    def load_from_file(cls, full_path: Path,
                       root_dir_overwrite: Path = None) -> SamplingAugDataset:
        tensors = torch.load(full_path)

        tensor_filename = full_path.stem
        meta_path: Path = full_path.parents[0] / f"{tensor_filename}_meta.pkl"
        if not meta_path.is_file():
            log.error(
                f"CustomTensorDataset: meta file belonging to {tensor_filename} can't be found")
            raise ValueError(
                f"CustomTensorDataset: meta file belonging to {tensor_filename} can't be found")
        with open(meta_path, 'rb') as meta_file:
            # for now, the img_paths still are OS dependent, so we sometimes struggle
            name, root_dir_string, img_paths = pickle.load(meta_file)
            # img_paths are relative so resolving these should always work
            img_paths = [Path(path_string) for path_string in img_paths]

        if root_dir_overwrite:
            root_dir = root_dir_overwrite
        else:
            root_dir = Path(root_dir_string)
            if not root_dir.is_dir():
                log.error(
                    f'CustomTensorDataset load(): Invalid root_dir "{root_dir_string}"'
                    f' found in metafile.'
                    f' Please provide `root_dir_overwrite` parameter.')
                sys.exit(-1)

        return SamplingAugDataset(name, tensors[0], tensors[1], root_dir=root_dir,
                                  img_paths=img_paths)

    def save_to_file(self, path: Path, description: str = "tensors"):
        torch.save(self.tensors, path / f"{self.name}_{description}.pt")
        # root dir and img ids are python primitives, should be easier like this
        # since I had some trouble loading the CustomTensorDataset with torch.load

        # make the Paths portable to other OS
        img_paths_strings = [str(path) for path in self.img_paths]
        root_dir_string = str(self.root_dir)

        with open(path / f"{self.name}_{description}_meta.pkl", 'wb') as meta_file:
            pickle.dump((self.name, root_dir_string, img_paths_strings), meta_file)


class ImageFolderToTensors(Step):
    @staticmethod
    def check_environment() -> Optional[str]:
        return None

    @classmethod
    def run(cls, state: Artifact, params: Config) -> Artifact:
        # TODO
        return Artifact()
