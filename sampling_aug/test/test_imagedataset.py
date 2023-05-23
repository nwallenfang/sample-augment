import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Subset
from torchvision.transforms import Normalize, Grayscale, ToTensor, Resize, Compose

from data.dataset import ImageDataset
from data.train_test_split import stratified_split
from utils.paths import project_path


def create_image_folder_dataset():
    preprocessing = Compose([
        Resize((256, 256)),
        ToTensor(),
        Grayscale(num_output_channels=3),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageDataset(project_path('data/gc-10'), transform=preprocessing)
    # dataset = GC10InMemoryDataset()
    train_dataset, test_dataset = stratified_split(dataset, train_ratio=0.8)

    # TODO apply labels from labels.json (only ~5 instances are mislabelled)

    visualize_class_distribution(dataset, train_dataset)
    visualize_class_distribution(dataset, test_dataset)


def visualize_class_distribution(complete_dataset: ImageDataset, subset: Subset):
    labels = np.array([complete_dataset.targets[idx] for idx in subset.indices])
    classes, counts = np.unique(labels, return_counts=True)
    print(counts)
    plt.figure()
    plt.bar(classes, counts)
    plt.show()


def test_loading_imagedataset():
    create_image_folder_dataset()
