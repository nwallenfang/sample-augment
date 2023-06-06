import os.path
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import Tensor
from torch.utils.data import TensorDataset
from torchvision.transforms import Resize, Compose, Grayscale, ToTensor
from tqdm import tqdm

from sample_augment.utils.log import log
from sample_augment.utils.paths import project_path


class ImageDataset(torchvision.datasets.ImageFolder):
    """
        Extension of ImageDataset from torchvision.
        Maybe later we'll need some custom behavior.
        Could for example also manage the secondary labels we have in GC-10.
    """
    pass


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
                       root_dir_overwrite: Path = None) -> "SamplingAugDataset":
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


def image_folder_to_tensor_dataset(image_dataset: ImageDataset,
                                   name: str = 'gc10',
                                   true_labels: dict[str, int] = None) -> SamplingAugDataset:
    """
        ImageFolder dataset is designed for big datasets that don't fit into RAM (think ImageNet).
        For GC10 we can easily load the whole dataset into RAM transform the ImageDataset into a
        Tensor-based one for this.
    """
    log.info('loading images into CustomTensorDataset..')
    # label_tensors = torch.tensor(image_dataset.targets, dtype=torch.int)
    # image_tensors = torch.stack([image_dataset[i][0] for i in tqdm(range(len(image_dataset)))])

    # save a dictionary pointing from the tensor indices to the image IDs / paths for traceability
    log.info('reading image paths (metadata)..')
    root_dir = Path(image_dataset.root)
    img_paths = []

    # to check for duplicates, since gc10 contains some duplicate images
    duplicate_dict = {}
    remove_these_idx = []
    removed_duplicates = 0

    for i, (img_path, _img_class) in tqdm(enumerate(image_dataset.imgs)):
        path_obj = Path(img_path)
        # FIXME this way of calculating the ids is gc10 specific..
        img_id = path_obj.stem[4:]

        if img_id not in duplicate_dict:
            duplicate_dict[img_id] = i
        else:
            # always take the first appearance of an image, might just need to change the class
            if true_labels:
                image_dataset.targets[i] = true_labels[img_id]
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

    # convert image_tensors to uint8 since that's the format needed for training on StyleGAN
    image_data: np.ndarray = image_tensors.numpy()

    # there could be negative values, so shift by the minimum to bring into range 0...max
    image_data -= np.min(image_data)
    # image_data might be float32, convert to float64 for higher accuracy when scaling
    image_data = image_data.astype(np.float64) / np.max(image_data)
    image_data *= 255  # Now scale by 255
    image_data = image_data.astype(np.uint8)
    image_tensors = torch.from_numpy(image_data)
    tensor_dataset = SamplingAugDataset(name, image_tensors, label_tensors, img_paths=img_paths,
                                        root_dir=root_dir)
    return tensor_dataset


def main():
    from sample_augment.data.train_test_split import create_train_val_test_sets
    """
        Runs the complete data processing pipeline.
        Load GC10 dataset, do label sanitization, do image preprocessing, do train/test/val split.
    """
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
    # tensor_path
    if not os.path.exists(project_path('data/interim/gc10_tensors.pt')):
        image_dataset = ImageDataset(project_path('data/gc-10'), transform=preprocessing)
        # TODO pass labels.json contents
        tensor_dataset: SamplingAugDataset = image_folder_to_tensor_dataset(image_dataset)
        del image_dataset
        assert isinstance(tensor_dataset, TensorDataset)
        dataset_dir = Path(project_path('data/interim/', create=True))
        tensor_dataset.save_to_file(dataset_dir)
    else:
        tensor_dataset = SamplingAugDataset.load_from_file(
            Path(project_path('data/interim/gc10_tensors.pt')))

    train_data, val_data, test_data = create_train_val_test_sets(tensor_dataset, random_seed=15)
    del tensor_dataset
    train_data.save_to_file(path=Path(project_path('data/interim/')), description='train')
    val_data.save_to_file(path=Path(project_path('data/interim/')), description='val')
    test_data.save_to_file(path=Path(project_path('data/interim/')), description='test')


def test_duplicate_ids():
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
    image_dataset = ImageDataset(project_path('data/gc-10-mini'), transform=preprocessing)
    tensor_dataset: SamplingAugDataset = image_folder_to_tensor_dataset(image_dataset)
    del image_dataset
    assert isinstance(tensor_dataset, TensorDataset)
    dataset_dir = Path(project_path('data/interim/', create=True))
    tensor_dataset.save_to_file(dataset_dir)


if __name__ == '__main__':
    # tensor_dataset: TensorDataset = torch.load(project_path('data/interim/gc10_tensors.pt'))
    # create_train_val_test_sets(tensor_dataset)
    main()