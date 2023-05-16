import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision.transforms import Resize, Compose, Grayscale, ToTensor, Normalize
from tqdm import tqdm

from sampling_aug.data.train_test_split import stratified_split
# from sampling_aug.data.gc10 import GC10InMemoryDataset
from utils.paths import project_path

import logging


class ImageDataset(torchvision.datasets.ImageFolder):
    """
        Extension of ImageDataset from torchvision.
        Maybe later we'll need some custom behavior.
        Could for example also manage the secondary labels we have in GC-10.
    """
    pass


def visualize_class_distribution(complete_dataset: ImageDataset, subset: Subset):
    labels = np.array([complete_dataset.targets[idx] for idx in subset.indices])
    classes, counts = np.unique(labels, return_counts=True)
    print(counts)
    plt.figure()
    plt.bar(classes, counts)
    plt.show()


def image_folder_to_tensor_dataset(image_dataset: ImageDataset):
    """
        ImageFolder dataset is designed for big datasets that don't fit into RAM (think ImageNet).
        For GC10 we can easily load the whole dataset into RAM transform the ImageDataset into a Tensor-based one
        for this.
    """
    logging.info('loading images into TensorDataset..')
    label_tensors = torch.tensor(image_dataset.targets, dtype=torch.int)
    image_tensors = torch.stack([image_dataset[i][0] for i in tqdm(range(len(image_dataset)))])
    # TODO save a dictionary pointing from the tensor indices to the image IDs / paths for traceability

    # convert image_tensors to uint8
    image_data: np.ndarray = image_tensors.numpy()

    # there could be negative values, so shift by the minimum to bring into range 0...max
    image_data -= np.min(image_data)
    # image_data might be float32, convert to float64 for higher accuracy when scaling
    image_data = image_data.astype(np.float64) / np.max(image_data)
    image_data *= 255  # Now scale by 255
    image_data = image_data.astype(np.uint8)
    image_tensors = torch.from_numpy(image_data)
    tensor_dataset = TensorDataset(image_tensors, label_tensors)
    return tensor_dataset


def test_image_folder_dataset():
    preprocessing = Compose([
        Resize((256, 256)),
        ToTensor(),
        Grayscale(num_output_channels=3),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageDataset(project_path('data/gc-10'), transform=preprocessing)
    # dataset = GC10InMemoryDataset()
    # TODO stratified test train split, maybe work with scikit-learn for this?
    train_dataset, test_dataset = stratified_split(dataset, train_ratio=0.8)
    # TODO apply labels from labels.json (only ~5 instances are mislabelled)

    visualize_class_distribution(dataset, train_dataset)
    visualize_class_distribution(dataset, test_dataset)
    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    for img_batch, labels in train_dataloader:
        print(img_batch.size())
        print(labels)


def main():
    # test_image_folder_dataset()
    preprocessing = Compose([
        Resize((256, 256)),
        ToTensor(),
        Grayscale(num_output_channels=3),
        # TODO check if these normalization factors are correct for GC-10
        # on second thought: These normalization factors might be used to bring GC10
        # to the same distribution as ImageNet since we're using a DenseNet classifier
        # that was pretrained on ImageNet
        # Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageDataset(project_path('data/gc-10'), transform=preprocessing)
    tensor_dataset = image_folder_to_tensor_dataset(dataset)

    dataset_dir = project_path('data/interim/', create=True)
    dataset_path = os.path.join(dataset_dir, 'gc10_tensors.pt')
    # tensor_dataset = torch.load(dataset_path)
    # print(len(tensor_dataset))
    # TODO I need the image ids as well in the dataset
    torch.save(tensor_dataset, dataset_path)


if __name__ == '__main__':
    main()
