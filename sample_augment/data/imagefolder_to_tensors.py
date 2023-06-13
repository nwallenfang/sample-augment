# from __future__ import annotations

import pickle
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torchvision
from torch import Tensor
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, Grayscale, ToTensor, Compose
from tqdm import tqdm

from sample_augment.core import step
from sample_augment.core.artifact import Artifact
from sample_augment.data.download_gc10 import GC10Folder
from sample_augment.utils import log


class SampleAugmentDataset(TensorDataset, Artifact):
    # TODO move DataSet class into its own script
    """
        PyTorch TensorDataset with the extension that we're also saving the paths to the
        original images.
    """
    name: str
    root_dir: Path
    # TODO verify that these are just the ids or else make them so
    img_ids: List[str]
    # redeclare these so they are turned into Pydantic fields
    # needed for proper serialization of this class
    # image_tensor: Tensor
    # label_tensor: Tensor
    tensors: Tuple[Tensor, ...]

    # noinspection PyMissingConstructor
    def __init__(self, name: str, image_tensor: Tensor, label_tensor: Tensor, img_ids: List[str],
                 root_dir: Path, *tensors: Tensor):
        """
        Args:
            img_ids (List[str]): list going from Tensor index to relative path of the given image
            root_dir (Path): for resolving the relative paths from img_ids
        """
        # skipping super constructor since it doesn't play well with our multi-inheritance
        assert len(image_tensor) == len(label_tensor)
        Artifact.__init__(self, name=name, root_dir=root_dir, img_ids=img_ids,
                          tensors=(image_tensor, label_tensor))
        # self.name = name
        # self.root_dir = root_dir
        # self.img_ids = img_ids
        # self.image_tensor = image_tensor
        # self.label_tensor = label_tensor
        # self.tensors = (image_tensor, label_tensor)

    @property
    def image_tensor(self):
        return self.tensors[0]

    @property
    def label_tensor(self):
        return self.tensors[1]

    def get_img_id(self, index: int) -> str:
        """
            img_id is just the filename without the file ending and the redundant `img_` prefix
        """
        # if isinstance(self.img_paths[index], Path):
        #     img_id = str(self.img_paths[index].name)
        #     img_id = img_id.split('.')[0][4:]

        img_id = self.img_ids[index]
        return str(img_id)

    def get_img_path(self, index: int) -> Path:
        return Path.joinpath(self.root_dir, self.img_ids[index])

    @classmethod
    def load_from_file(cls, full_path: Path,
                       root_dir_overwrite: Path = None) -> "SampleAugmentDataset":
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

        return SampleAugmentDataset(name, tensors[0], tensors[1], root_dir=root_dir,
                                    img_ids=img_paths)

    def save_to_file(self, path: Path, description: str = "tensors"):
        torch.save(self.tensors, path / f"{self.name}_{description}.pt")
        # root dir and img ids are python primitives, should be easier like this
        # since I had some trouble loading the CustomTensorDataset with torch.load

        # make the Paths portable to other OS
        img_paths_strings = [str(path) for path in self.img_ids]
        root_dir_string = str(self.root_dir)

        with open(path / f"{self.name}_{description}_meta.pkl", 'wb') as meta_file:
            pickle.dump((self.name, root_dir_string, img_paths_strings), meta_file)


class ImageFolderPath(Artifact):
    image_folder_path: Path


@step
def gc10_adapter(gc10_data: GC10Folder) -> ImageFolderPath:
    return ImageFolderPath(image_folder_path=gc10_data.image_folder_path)


# meta: don't really like having to create an artifact for single attributes but fine
@step(name="ImageFolderToTensors")
def imagefolder_to_tensors(image_folder_path: ImageFolderPath,
                           name: str,  # true_labels: dict[str, int] = None
                           ) -> SampleAugmentDataset:
    """
        ImageFolder dataset is designed for big datasets that don't fit into RAM (think ImageNet).
        For GC10 we can easily load the whole dataset into RAM transform the ImageDataset into a
        Tensor-based one for this.
    """
    log.info("Creating Pytorch ImageFolder")
    preprocessing = Compose([
        Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
               antialias=True),
        ToTensor(),
        Grayscale(num_output_channels=3),
        # These normalization factors might be used to bring our dataset (e.g. GC10)
        # to the same distribution as ImageNet since we're using a
        # DenseNet classifier that was pretrained on ImageNet.
        # Optimally, the Generator should generate images with this distribution as well.
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_dataset = ImageFolder(str(image_folder_path.image_folder_path), transform=preprocessing)

    log.info('Converting ImageFolder to  SampleAugmentDataset..')
    # label_tensors = torch.tensor(image_dataset.targets, dtype=torch.int)
    # image_tensors = torch.stack([image_dataset[i][0] for i in tqdm(range(len(image_dataset)))])

    # save a dictionary pointing from the tensor indices to the image IDs / paths for traceability
    log.info('Reading image paths (metadata)..')
    root_dir = Path(image_dataset.root)
    img_paths = []
    img_ids = []

    # to check for duplicates, since gc10 contains some duplicate images
    duplicate_dict = {}
    remove_these_idx = []
    removed_duplicates = 0

    for i, (img_path, _img_class) in tqdm(enumerate(image_dataset.imgs)):
        path_obj = Path(img_path)
        # FIXME this way of calculating the ids is gc10 specific..
        img_id = path_obj.stem[4:]
        img_ids.append(img_id)

        if img_id not in duplicate_dict:
            duplicate_dict[img_id] = i
        else:
            # always take the first appearance of an image, might just need to change the class
            # if true_labels:
            #     image_dataset.targets[i] = true_labels[img_id]
            remove_these_idx.append(i)
            removed_duplicates += 1
            continue

        img_paths.append(path_obj.relative_to(root_dir))

    if removed_duplicates > 0:
        log.warning(f'Removed {removed_duplicates} duplicates from the dataset.')

    # filter duplicates
    unique_indices = list(set(range(len(image_dataset))) - set(remove_these_idx))
    image_tensors = torch.stack([image_dataset[i][0] for i in tqdm(unique_indices)])
    label_tensors = torch.tensor([image_dataset.targets[i] for i in tqdm(unique_indices)],
                                 dtype=torch.int)
    img_ids = [img_ids[idx] for idx in unique_indices]

    # convert image_tensors to uint8 since that's the format needed for training on StyleGAN
    image_data: np.ndarray = image_tensors.numpy()

    # there could be negative values, so shift by the minimum to bring into range 0...max
    image_data -= np.min(image_data)
    # image_data might be float32, convert to float64 for higher accuracy when scaling
    image_data = image_data.astype(np.float64) / np.max(image_data)
    image_data *= 255  # Now scale by 255
    image_data = image_data.astype(np.uint8)
    image_tensors = torch.from_numpy(image_data)
    # TODO test that new img ids are correct
    tensor_dataset = SampleAugmentDataset(name=name, image_tensor=image_tensors, label_tensor=label_tensors,
                                          img_ids=img_ids,
                                          root_dir=root_dir)
    return tensor_dataset
