import pprint
import sys
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import torchvision
from pydantic import validator
from torch import Tensor
from torch.utils.data import TensorDataset
from torchvision.datasets import ImageFolder
from torchvision.transforms import Resize, Compose, Grayscale, ToTensor
from tqdm import tqdm

from sample_augment.core import Artifact, step
from sample_augment.data.gc10.download_gc10 import GC10Folder
from sample_augment.data.gc10.read_labels import SanitizedGC10Labels
from sample_augment.utils.log import log


class ImageDataset(torchvision.datasets.ImageFolder):
    """
        Extension of ImageDataset from torchvision.
        Maybe later we'll need some custom behavior.
        Could for example also manage the secondary labels we have in GC-10.
    """
    pass


class AugmentDataset(TensorDataset, Artifact):
    # TODO rename to CompleteDataset
    """
        PyTorch TensorDataset with the extension that we're also saving the paths to the
        original images.
    """
    name: str
    root_dir: Path
    img_ids: List[str]
    # redeclare these so they are turned into Pydantic fields
    # needed for proper serialization of this class
    tensors: Tuple[Tensor, ...]

    primary_label_tensor: Tensor

    def __init__(self, name: str, tensors: Tuple[Tensor, Tensor], img_ids: List[str],
                 root_dir: Path, **kwargs):
        """
        Args:
            img_ids (List[str]): list going from Tensor index to relative path of the given image
            root_dir (Path): for resolving the relative paths from img_ids
        """
        assert len(tensors[0]) == len(tensors[1]), "Image tensor length doesn't match label tensor length!"
        Artifact.__init__(self, name=name, root_dir=root_dir, img_ids=img_ids,
                          tensors=tensors, **kwargs)

    def subset(self, indices: Union[List[int], np.ndarray], name: str = None) -> 'AugmentDataset':
        # copies the tensors, so this is a copy rather than a view
        # so potential for optimization, which we will ignore for now.
        subset_tensors = self.tensors[0][indices], self.tensors[1][indices]  # : Tuple[Tensor, Tensor]
        primary_subset = self.primary_label_tensor[indices]
        subset_img_ids = [self.img_ids[i] for i in indices]

        return AugmentDataset(
            name=name if name else "subset",
            root_dir=self.root_dir,
            img_ids=subset_img_ids,
            tensors=subset_tensors,
            primary_label_tensor=primary_subset
        )

    @classmethod
    def from_existing(cls, existing_dataset: 'AugmentDataset', name=None) -> 'AugmentDataset':
        """
            'copy-constructor' used for constructing subclasses such as TestSet.
        """

        new_dataset = cls(name=name if name else existing_dataset.name, tensors=(existing_dataset.tensors[0],
                                                                                 existing_dataset.tensors[1]),
                          img_ids=existing_dataset.img_ids,
                          root_dir=existing_dataset.root_dir,
                          primary_label_tensor=existing_dataset.primary_label_tensor)
        return new_dataset

    @property
    def image_tensor(self):
        return self.tensors[0]

    @property
    def label_tensor(self):
        return self.tensors[1]

    @validator("tensors")
    def validate_image_tensor_dtype(cls, value):
        if not value[0].dtype == torch.uint8:
            raise ValueError(f"invalid image tensor dtype {value[0].dtype}, expecting uint8")
        else:
            return value

    @property
    def num_classes(self):
        # assumption: multilabel
        return len(self.tensors[1][0])  # assuming all label vectors have the same length

    # @property
    # def transform(self):
    #     return self._transform
    #
    # @transform.setter
    # def transform(self, transform):
    #     self._transform = transform


class ImageFolderPath(Artifact):
    image_dir: Path


@step
def gc10_adapter(gc10_data: GC10Folder) -> ImageFolderPath:
    return ImageFolderPath(image_dir=gc10_data.image_dir)


def create_multilabel_tensor(labels, ids, num_classes=10):
    multilabel_tensor = torch.zeros((len(ids), num_classes), dtype=torch.int)
    for i, img_id in enumerate(ids):
        primary_class = labels[img_id]['y']
        secondary_classes = labels[img_id]['secondary']
        multilabel_tensor[i, primary_class] = 1
        for class_ in secondary_classes:
            multilabel_tensor[i, class_] = 1
    return multilabel_tensor


# meta: don't really like having to create an artifact for single attributes but fine
@step
def imagefolder_to_tensors(image_folder_path: ImageFolderPath, sanitized_labels: SanitizedGC10Labels) -> AugmentDataset:
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
    remove_info = []
    removed_duplicates = 0

    for i, (img_path, _img_class) in tqdm(enumerate(image_dataset.imgs), file=sys.stdout):
        path_obj = Path(img_path)
        # gc10 specific way of calculating img_id !
        img_id = path_obj.stem[4:]
        img_ids.append(img_id)

        if img_id not in duplicate_dict:
            duplicate_dict[img_id] = i
        else:
            # always take the first appearance of an image, might just need to change the class
            # if true_labels:
            #     image_dataset.targets[i] = true_labels[img_id]
            remove_these_idx.append(i)
            remove_info.append((img_id, _img_class))
            removed_duplicates += 1
            continue

        img_paths.append(path_obj.relative_to(root_dir))

    if removed_duplicates > 0:
        log.warning(f'Removed {removed_duplicates} duplicates from the dataset.')
        log.info(f'Duplicates: {pprint.pformat(remove_info)}')

    # filter duplicates
    unique_indices = list(set(range(len(image_dataset))) - set(remove_these_idx))
    image_tensor = torch.stack([image_dataset[i][0] for i in tqdm(unique_indices, file=sys.stdout)])
    # label_tensor = torch.tensor([image_dataset.targets[i] for i in unique_indices],
    #                             dtype=torch.int)
    single_label_tensor = torch.tensor([sanitized_labels.labels[img_ids[i]]['y'] for i in unique_indices],
                                       dtype=torch.int)
    img_ids = [img_ids[idx] for idx in unique_indices]
    multi_label_tensor = create_multilabel_tensor(sanitized_labels.labels, img_ids)

    class_counts = torch.sum(multi_label_tensor, dim=0)

    # Print the class counts
    for i, count in enumerate(class_counts):
        log.info(f"Class {sanitized_labels.class_names[i]}: {count}")
    # class_counts = torch.bincount(label_tensor)
    # log.info(f"counts: {class_counts}")

    # convert image_tensors to uint8 since that's the format needed for training on StyleGAN
    image_data: np.ndarray = image_tensor.numpy()

    # there could be negative values, so shift by the minimum to bring into range 0...max
    image_data -= np.min(image_data)
    # image_data might be float32, convert to float64 for higher accuracy when scaling
    image_data = image_data.astype(np.float64) / np.max(image_data)
    image_data *= 255  # Now scale by 255
    image_data = image_data.astype(np.uint8)
    image_tensor = torch.from_numpy(image_data)

    tensor_dataset = AugmentDataset(name="complete", tensors=(image_tensor, multi_label_tensor),
                                    primary_label_tensor=single_label_tensor,
                                    img_ids=img_ids,
                                    root_dir=root_dir)
    return tensor_dataset
