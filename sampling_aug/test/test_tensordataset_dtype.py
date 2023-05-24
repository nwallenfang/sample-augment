from pathlib import Path

import torch
from matplotlib import pyplot as plt

from data.dataset import CustomTensorDataset
from utils.paths import project_path


def test_tensor_dataset():
    dataset = torch.load(project_path('data/interim/gc10_tensors.pt'))
    img = dataset.tensors[0][0].permute(1, 2, 0)
    print(img)
    plt.imshow(img)
    plt.show()


def test_custom_tensor_dataset():
    path = Path(project_path('data/interim/'))
    dataset: CustomTensorDataset = CustomTensorDataset.load(path, 'gc10')
    print()
    print(f"Full path: {dataset.get_img_path(10)}")
    print(f"Img ID: {dataset.get_img_id(10)}")




