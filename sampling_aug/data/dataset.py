import os.path
import pickle
from pathlib import Path

import numpy as np
import torch
import torchvision
from torch import Tensor
from torch.utils.data import TensorDataset
from torchvision.transforms import Resize, Compose, Grayscale, ToTensor
from tqdm import tqdm

from utils.logging import logger
from utils.paths import project_path


class ImageDataset(torchvision.datasets.ImageFolder):
    """
        Extension of ImageDataset from torchvision.
        Maybe later we'll need some custom behavior.
        Could for example also manage the secondary labels we have in GC-10.
    """
    pass


class CustomTensorDataset(TensorDataset):
    """
        PyTorch TensorDataset with the extension that we're also saving the paths to the original images
    """

    def __init__(self, name: str, image_tensor: Tensor, label_tensor: Tensor, img_paths: list, root_dir: Path):
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
        img_id = str(self.img_paths[index].name)
        img_id = img_id.split('.')[0][4:]
        return str(img_id)

    def get_img_path(self, index: int) -> Path:
        return Path.joinpath(self.root_dir, self.img_paths[index])

    @classmethod
    def load(cls, full_path: Path) -> "CustomTensorDataset":
        tensors = torch.load(full_path)

        tensor_filename = full_path.stem
        meta_path: Path = full_path.parents[0] / f"{tensor_filename}_meta.pkl"
        if not meta_path.is_file():
            logger.error(f"CustomTensorDataset: meta file belonging to {tensor_filename} can't be found")
            raise ValueError(f"CustomTensorDataset: meta file belonging to {tensor_filename} can't be found")
        with open(meta_path, 'rb') as meta_file:
            name, root_dir, img_paths = pickle.load(meta_file)

        return CustomTensorDataset(name, tensors[0], tensors[1], root_dir=root_dir, img_paths=img_paths)

    def save(self, path: Path, description: str = "tensors"):
        torch.save(self.tensors, path / f"{self.name}_{description}.pt")
        # root dir and img ids are python primitives, should be easier like this
        # since I had some trouble loading the CustomTensorDataset with torch.load
        # TODO it's bad to save those file paths because they're not portable at all
        # could be possible to only save the ids..
        with open(path / f"{self.name}_{description}_meta.pkl", 'wb') as meta_file:
            pickle.dump((self.name, self.root_dir, self.img_paths), meta_file)


def image_folder_to_tensor_dataset(image_dataset: ImageDataset, name: str = 'gc10') -> CustomTensorDataset:
    """
        ImageFolder dataset is designed for big datasets that don't fit into RAM (think ImageNet).
        For GC10 we can easily load the whole dataset into RAM transform the ImageDataset into a Tensor-based one
        for this.
    """
    logger.info('loading images into CustomTensorDataset..')
    label_tensors = torch.tensor(image_dataset.targets, dtype=torch.int)
    image_tensors = torch.stack([image_dataset[i][0] for i in tqdm(range(len(image_dataset)))])

    # save a dictionary pointing from the tensor indices to the image IDs / paths for traceability
    logger.info('reading image paths (metadata)..')
    root_dir = Path(project_path('data/gc-10'))  # FIXME make independent of gc10 (ImageDataSet might contain the path)
    img_paths = []
    for img_path, _img_class in tqdm(image_dataset.imgs):
        path_obj = Path(img_path)
        img_paths.append(path_obj.relative_to(root_dir))

    # convert image_tensors to uint8 since that's the format needed for training on StyleGAN
    image_data: np.ndarray = image_tensors.numpy()

    # there could be negative values, so shift by the minimum to bring into range 0...max
    image_data -= np.min(image_data)
    # image_data might be float32, convert to float64 for higher accuracy when scaling
    image_data = image_data.astype(np.float64) / np.max(image_data)
    image_data *= 255  # Now scale by 255
    image_data = image_data.astype(np.uint8)
    image_tensors = torch.from_numpy(image_data)
    tensor_dataset = CustomTensorDataset(name, image_tensors, label_tensors, img_paths=img_paths, root_dir=root_dir)
    return tensor_dataset


def main():
    from data.train_test_split import create_train_val_test_sets
    """
        Runs the complete data processing pipeline.
        Load GC10 dataset, do label sanitization, do image preprocessing, do train/test/val split.
    """
    preprocessing = Compose([
        Resize((256, 256), interpolation=torchvision.transforms.InterpolationMode.BICUBIC, antialias=True),
        ToTensor(),
        Grayscale(num_output_channels=3),
        # These normalization factors might be used to bring GC10
        # to the same distribution as ImageNet since we're using a DenseNet classifier that was pretrained on ImageNet.
        # Optimally, the Generator should generate images with this distribution as well.
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # tensor_path
    if not os.path.exists(project_path('data/interim/gc10_tensors.pt')):
        image_dataset = ImageDataset(project_path('data/gc-10'), transform=preprocessing)
        tensor_dataset: CustomTensorDataset = image_folder_to_tensor_dataset(image_dataset)
        del image_dataset
        assert isinstance(tensor_dataset, TensorDataset)
        dataset_dir = Path(project_path('data/interim/', create=True))
        tensor_dataset.save(dataset_dir)
    else:
        tensor_dataset = CustomTensorDataset.load(Path(project_path('data/interim/gc10_tensors.pt')))

    train_data, val_data, test_data = create_train_val_test_sets(tensor_dataset)
    del tensor_dataset
    train_data.save(path=Path(project_path('data/interim/')), description='train')
    val_data.save(path=Path(project_path('data/interim/')), description='val')
    test_data.save(path=Path(project_path('data/interim/')), description='test')


if __name__ == '__main__':
    # tensor_dataset: TensorDataset = torch.load(project_path('data/interim/gc10_tensors.pt'))
    # create_train_val_test_sets(tensor_dataset)
    main()
