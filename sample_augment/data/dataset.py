import sys
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torchvision
from torch import Tensor
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, Compose, Grayscale, ToTensor
from tqdm import tqdm

from sample_augment.core import Artifact, step
from sample_augment.data.gc10.download_gc10 import GC10Folder
from sample_augment.utils.log import log
from sample_augment.utils.paths import project_path


class ImageDataset(torchvision.datasets.ImageFolder):
    """
        Extension of ImageDataset from torchvision.
        Maybe later we'll need some custom behavior.
        Could for example also manage the secondary labels we have in GC-10.
    """
    pass


class AugmentDataset(TensorDataset, Artifact):
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
    tensors: Tuple[Tensor, ...]

    # noinspection PyMissingConstructor
    def __init__(self, name: str, tensors: Tuple[Tensor, Tensor], img_ids: List[str],
                 root_dir: Path):
        """
        Args:
            img_ids (List[str]): list going from Tensor index to relative path of the given image
            root_dir (Path): for resolving the relative paths from img_ids
        """
        # assert number of images == number of labels
        assert len(tensors[0]) == len(tensors[1])
        Artifact.__init__(self, name=name, root_dir=root_dir, img_ids=img_ids,
                          tensors=tensors)

    def subset(self, indices: Union[List[int], np.ndarray]) -> 'AugmentDataset':
        # copies the tensors, so this is a copy rather than a view
        # so potential for optimization, which we will ignore for now.
        subset_tensors: Tuple[Tensor, Tensor] = self.tensors[0][indices], self.tensors[1][indices]
        subset_img_ids = [self.img_ids[i] for i in indices]

        return AugmentDataset(
            name=self.name,
            root_dir=self.root_dir,
            img_ids=subset_img_ids,
            tensors=subset_tensors
        )

    @property
    def image_tensor(self):
        return self.tensors[0]

    @property
    def label_tensor(self):
        return self.tensors[1]


def test_train_test_split():
    # TODO next stop: train, test split :)
    # train_data, val_data, test_data = create_train_val_test_sets(tensor_dataset, random_seed=15)
    # train_data.save_to_file(path=Path(project_path('data/interim/')), description='train')
    # val_data.save_to_file(path=Path(project_path('data/interim/')), description='val')
    # test_data.save_to_file(path=Path(project_path('data/interim/')), description='test')
    pass


def test_duplicate_ids():
    # TODO fit test into the new framework
    preprocessing = Compose([
        Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
               antialias=True),
        ToTensor(),
        Grayscale(num_output_channels=3),
        # These normalization factors might be used to bring GC10
        # to the same distribution as ImageNet since we're using a
        # DenseNet classifier that was pretrained on ImageNet.
        # Optimally, the Generator should generate images with this distribution as well.
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_dataset = ImageDataset(project_path('data/gc10-mini'), transform=preprocessing)
    # tensor_dataset: AugmentDataset = image_folder_to_tensor_dataset(image_dataset)
    # del image_dataset
    # assert isinstance(tensor_dataset, TensorDataset)
    dataset_dir = Path(project_path('data/interim/', create=True))
    # tensor_dataset.save_to_file(dataset_dir)


class ImageFolderPath(Artifact):
    image_dir: Path


@step
def gc10_adapter(gc10_data: GC10Folder) -> ImageFolderPath:
    return ImageFolderPath(image_dir=gc10_data.image_dir)


# meta: don't really like having to create an artifact for single attributes but fine
@step()
def imagefolder_to_tensors(image_folder_path: ImageFolderPath,
                           name: str,  # true_labels: dict[str, int] = None
                           ) -> AugmentDataset:
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
    image_dataset = ImageFolder(str(image_folder_path.image_dir), transform=preprocessing)

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

    for i, (img_path, _img_class) in tqdm(enumerate(image_dataset.imgs), file=sys.stdout):
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
    image_tensor = torch.stack([image_dataset[i][0] for i in tqdm(unique_indices, file=sys.stdout)])
    label_tensor = torch.tensor([image_dataset.targets[i] for i in unique_indices],
                                dtype=torch.int)
    img_ids = [img_ids[idx] for idx in unique_indices]

    # convert image_tensors to uint8 since that's the format needed for training on StyleGAN
    image_data: np.ndarray = image_tensor.numpy()

    # there could be negative values, so shift by the minimum to bring into range 0...max
    image_data -= np.min(image_data)
    # image_data might be float32, convert to float64 for higher accuracy when scaling
    image_data = image_data.astype(np.float64) / np.max(image_data)
    image_data *= 255  # Now scale by 255
    image_data = image_data.astype(np.uint8)
    image_tensor = torch.from_numpy(image_data)
    # TODO test that new img ids are correct
    tensor_dataset = AugmentDataset(name=name, tensors=(image_tensor, label_tensor),
                                    img_ids=img_ids,
                                    root_dir=root_dir)
    return tensor_dataset
