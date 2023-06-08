import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
# noinspection PyPep8Naming
import torchvision.transforms.functional as F
from torch import Tensor
from torch.utils.data import Subset
from torchvision.transforms import Normalize, Grayscale, ToTensor, Resize, Compose
from torchvision.utils import make_grid

from sample_augment.data.dataset import ImageDataset
from sample_augment.steps.imagefolder_to_tensors import SamplingAugDataset
from sample_augment.data.train_test_split import stratified_split
from sample_augment.utils.paths import project_path
from sample_augment.data import dataset as dataset_package


def visualize_class_distribution(complete_dataset: ImageDataset, subset: Subset):
    labels = np.array([complete_dataset.targets[idx] for idx in subset.indices])
    classes, counts = np.unique(labels, return_counts=True)
    print(counts)
    plt.figure()
    plt.bar(classes, counts)
    plt.show()


def create_image_folder_dataset():
    preprocessing = Compose([
        Resize((256, 256)),
        ToTensor(),
        Grayscale(num_output_channels=3),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    dataset = ImageDataset(project_path('data/gc10'), transform=preprocessing)
    # dataset = GC10InMemoryDataset()
    train_dataset, test_dataset = stratified_split(dataset, train_ratio=0.8)

    # TODO apply labels from labels.json (only ~5 instances are mislabelled)

    visualize_class_distribution(dataset, train_dataset)
    visualize_class_distribution(dataset, test_dataset)

    return dataset


def test_uniqueness_of_ids():
    complete = SamplingAugDataset.load_from_file(Path(project_path('data/interim/gc10_tensors.pt')))
    all_ids = [complete.get_img_id(i) for i in range(len(complete))]
    duplicate_ids = set([x for x in all_ids if all_ids.count(x) > 1])
    print()
    print(duplicate_ids)
    assert duplicate_ids == []


def test_train_test_split_load_gc10():
    if not os.path.exists(project_path('data/interim/gc10_tensors.pt')):
        print('initializing DataSets for this test..')
        dataset_package.main()

    # stylegan training doesn't work after the new train test split
    # might be our dataset or because I'm trying to do transfer learning with a FFHQ checkpoint.
    train = SamplingAugDataset.load_from_file(Path(project_path('data/interim/gc10_train.pt')))
    complete = SamplingAugDataset.load_from_file(Path(project_path('data/interim/gc10_tensors.pt')))
    n_complete = len(complete)
    del complete
    test = SamplingAugDataset.load_from_file(Path(project_path('data/interim/gc10_test.pt')))
    val = SamplingAugDataset.load_from_file(Path(project_path('data/interim/gc10_val.pt')))
    n_train, n_test, n_val = len(train), len(test), len(val)

    assert n_train + n_test + n_val == n_complete

    test_ids = set(test.get_img_id(i) for i in range(len(test)))
    train_ids = set(train.get_img_id(i) for i in range(len(train)))
    val_ids = set(val.get_img_id(i) for i in range(len(val)))

    test_train_intersect = test_ids.intersection(train_ids)
    print(test_train_intersect)
    test_val_intersect = test_ids.intersection(val_ids)
    print(test_val_intersect)
    train_val_intersect = train_ids.intersection(val_ids)
    print(train_val_intersect)
    # assert that the three sets are disjoint
    # for that the intersection between the sets should be the empty set
    assert test_train_intersect == set()
    assert test_val_intersect == set()
    assert train_val_intersect == set()

    # check for class balance in the stratified split
    train_labels = np.array(train.tensors[1])
    classes, counts = np.unique(train_labels, return_counts=True)
    plt.bar(classes, counts)
    plt.title('train class distribution')
    plt.show()

    test_labels = np.array(test.tensors[1])
    classes, counts = np.unique(test_labels, return_counts=True)
    plt.figure()
    plt.bar(classes, counts)
    plt.title('test class distribution')
    plt.show()

    # visually, the class distributions look similar, though they don't seem to follow
    # the "true distribution" found in the complete dataset
    # look at the complete distribution:
    # TODO ! class distribution test !

    example_img: Tensor = train[10][0]

    print()
    print(f"path: {train.get_img_path(10)}")
    print(f"shape: {example_img.size()}")
    # plt.imshow(example_img.permute(1, 2, 0))
    # plt.show()


def show(imgs):
    plt.rcParams["savefig.bbox"] = 'tight'

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
    # make a grid for each class, image IDs could be useful as well,
    # to compare to the unscaled/quantized version
    dataset = SamplingAugDataset.load_from_file(Path(project_path('data/interim/gc10_train.pt')))
    _data, _labels = dataset.tensors
    # TODO visual inspection
    for class_idx in range(10):
        pass
    grid = make_grid()
    show(grid)
    pass


if __name__ == '__main__':
    main()
