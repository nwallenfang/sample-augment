from pathlib import Path

import torch
from matplotlib import pyplot as plt

from sample_augment.steps.imagefolder_to_tensors import SamplingAugDataset
from sample_augment.utils.paths import project_path


def test_tensor_dataset():
    dataset = torch.load(project_path('data/interim/gc10_tensors.pt'))
    img = dataset.tensors[0][0].permute(1, 2, 0)
    print(img)
    plt.imshow(img)
    plt.show()


def test_custom_tensor_dataset():
    path = Path(project_path('data/interim/'))
    dataset: SamplingAugDataset = SamplingAugDataset.load_from_file(path)
    print()
    print(f"Full path: {dataset.get_img_path(10)}")
    print(f"Img ID: {dataset.get_img_id(10)}")




