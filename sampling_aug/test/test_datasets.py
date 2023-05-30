import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms.functional as F
from torch import Tensor
from torch.utils.data import Subset
from torchvision.transforms import Normalize, Grayscale, ToTensor, Resize, Compose
from torchvision.utils import make_grid

from data.dataset import ImageDataset, CustomTensorDataset
from data.train_test_split import stratified_split
from utils.paths import project_path
from data import dataset as dataset_package


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

    return dataset


def visualize_class_distribution(complete_dataset: ImageDataset, subset: Subset):
    labels = np.array([complete_dataset.targets[idx] for idx in subset.indices])
    classes, counts = np.unique(labels, return_counts=True)
    print(counts)
    plt.figure()
    plt.bar(classes, counts)
    plt.show()


def test_loading_imagedataset():
    create_image_folder_dataset()


def test_load_training_set():
    if not os.path.exists(project_path('data/interim/gc10_tensors.pt')):
        print('initializing DataSets for this test..')
        dataset_package.main()

    # stylegan training doesn't work after the new train test split
    # might be our dataset or because I'm trying to do transfer learning with a FFHQ checkpoint.
    train = CustomTensorDataset.load(Path(project_path('data/interim/gc10_train.pt')))
    complete = CustomTensorDataset.load(Path(project_path('data/interim/gc10_tensors.pt')))
    n_complete = len(complete)
    del complete
    test = CustomTensorDataset.load(Path(project_path('data/interim/gc10_test.pt')))
    val = CustomTensorDataset.load(Path(project_path('data/interim/gc10_val.pt')))
    n_train, n_test, n_val = len(train), len(test), len(val)
    del test
    del val

    assert n_train + n_test + n_val == n_complete
    print()
    # TODO assert that the sets are disjoint

    example_img: Tensor = train[10][0]
    _example_label = train[10][1]

    print(f"path: {train.get_img_path(10)}")
    print(f"shape: {example_img.size()}")
    plt.imshow(example_img.permute(1, 2, 0))
    plt.show()


plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


def main():
    """
        visual inspection of some images from the TensorDataset
    """
    # make a grid for each class, image IDs could be useful as well, to compare to the unscaled/quantized version
    dataset = CustomTensorDataset.load(Path(project_path('data/interim/gc10_train.pt')))
    data, labels = dataset.tensors
    for class_idx in range(10):
        pass
    grid = make_grid()
    show(grid)
    pass


if __name__ == '__main__':
    main()
