

from typing import Any, Callable, Optional
from matplotlib import pyplot as plt
import torch
from torch.utils.data import DataLoader
from torchvision.datasets.folder import default_loader
from torchvision.transforms import Resize, ToTensor, Grayscale, Normalize, Compose
import torchvision

from sampling_aug.utils.paths import resolve_project_path


class ImageDataset(torchvision.datasets.ImageFolder):
    """
        Extension of ImageDataset from torchvision.
        Maybe later we'll need some custom behavior.
        Could for example also manage the secondary labels we have in GC-10.
    """    
    pass

class GC10Dataset(ImageDataset):
    # TODO put in its own file
    def __init__(self, target_transform: Callable[..., Any] | None = None, loader: Callable[[str], Any] = ..., is_valid_file: Callable[[str], bool] | None = None):
        preprocessing_transform = Compose([
        Resize((256, 256)),
        ToTensor(),
        Grayscale(num_output_channels=3),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
        root = resolve_project_path('data/gc-10')
        super().__init__(root, preprocessing, target_transform, loader, is_valid_file)


def main():
    preprocessing = Compose([
        Resize((256, 256)),
        ToTensor(),
        Grayscale(num_output_channels=3),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_dataset = ImageDataset(resolve_project_path('data/gc-10'), transform=preprocessing)
    # had to move 'lable' subdirectory
    # TODO stratifed test train split, maybe work with scikit-learn for this?
    # TODO apply labels from labels.json (only ~5 instances are mislabelled)
    # TODO make images grayscale


    train_dataloader = DataLoader(image_dataset, batch_size=64, shuffle=True)

    for img_batch, labels in train_dataloader:
        print(img_batch.size())
        print(labels)
    


if __name__ == '__main__':
    main()