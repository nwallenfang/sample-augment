"""
    Perform a stratified train test split (balance classes).

"""
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, Subset, TensorDataset
from torchvision.datasets import ImageFolder


def stratified_split(dataset, train_ratio: float = 0.8, random_seed=42):
    """
        dataset can be instance of ImageFolder or TensorDataset.
        (can't use typing for Python3.7 compatibility..)
        perform a random stratified split of a dataset into two datasets
    """
    n = len(dataset)
    if type(dataset) is ImageFolder:
        labels = dataset.targets
    else:  # is TensorDataset
        labels = dataset.tensors[1]
    indices = list(range(n))

    # use scikit-learn stratified split on the indices
    train_indices, test_indices = train_test_split(indices, train_size=train_ratio,
                                                   stratify=labels, random_state=random_seed)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset
