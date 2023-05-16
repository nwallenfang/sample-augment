import torch
from matplotlib import pyplot as plt

from utils.paths import project_path


def test_tensor_dataset():
    dataset = torch.load(project_path('data/interim/gc10_tensors.pt'))
    img = dataset.tensors[0][0].permute(1, 2, 0)
    print(img)
    plt.imshow(img)
    plt.show()
