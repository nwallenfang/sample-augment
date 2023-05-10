import torchvision
from torch.utils.data import DataLoader, Dataset

from sampling_aug.data.gc10 import GC10InMemoryDataset

class ImageDataset(torchvision.datasets.ImageFolder):
    """
        Extension of ImageDataset from torchvision.
        Maybe later we'll need some custom behavior.
        Could for example also manage the secondary labels we have in GC-10.
    """
    pass


def main():
    # dataset = ImageDataset(resolve_project_path('data/gc-10'), transform=preprocessing)
    dataset = GC10InMemoryDataset()
    # TODO stratified test train split, maybe work with scikit-learn for this?
    # TODO apply labels from labels.json (only ~5 instances are mislabelled)

    train_dataloader = DataLoader(dataset, batch_size=64, shuffle=True)

    for img_batch, labels in train_dataloader:
        print(img_batch.size())
        print(labels)


class GC10InMemory(Dataset):
    def __init__(self):
        pass


if __name__ == '__main__':
    main()
