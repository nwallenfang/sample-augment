"""
    Perform a stratified train test split (balance classes).

"""
import os
from logging import error

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch import LongTensor
from torch.utils.data import Subset, TensorDataset
from torchvision.datasets import ImageFolder

from utils.paths import project_path


def stratified_split(dataset, train_ratio: float = 0.8, random_seed: int = 42, min_instances_per_class: int = 10):
    """
        dataset can be an instance of ImageFolder or TensorDataset.
        (can't use typing for Python3.7 compatibility..)
        perform a random stratified split of a dataset into two datasets
    """
    np.random.seed(random_seed)
    # TODO assert this is deterministic if random_seed is set
    n = len(dataset)
    if type(dataset) is ImageFolder:
        labels = dataset.targets
    else:  # is TensorDataset
        labels = dataset.tensors[1]
    indices = list(range(n))

    # use scikit-learn stratified split on the indices
    train_indices, test_indices = train_test_split(indices, train_size=train_ratio,
                                                   stratify=labels, random_state=random_seed)
    train_indices, test_indices = np.array(train_indices), np.array(test_indices)
    # calculate instance counts per class of the test set, shift instances from the training set to ensure min count
    class_counts = np.bincount(labels[test_indices])
    for class_no in np.where(class_counts < min_instances_per_class):
        class_no = class_no.item()
        number_of_instances = min_instances_per_class - class_counts[class_no]
        # randomly choose n_missing instances from train to put into test set
        # class_indices is an array of indices into train_indices, which all have the label of the needed class
        class_indices = np.argwhere(labels[train_indices] == class_no).flatten()
        if len(class_indices) < number_of_instances:
            error(f'stratified split: Not enough instances of class with label {class_no} to fulfill '
                  f'min_instances={min_instances_per_class}')
            continue
        # choose the indices that will be moved from train to test randomly
        indices_to_move = np.random.choice(class_indices, number_of_instances, replace=False)
        # actual indices of the instances (labels tensor) that will be moved
        # we're working on indices of indices (since it's a Subset), which can be confusing!
        instances_to_move = train_indices[indices_to_move]

        test_indices = np.append(test_indices, instances_to_move)
        train_indices = np.delete(train_indices, indices_to_move)

    train_dataset = Subset(dataset, train_indices)
    test_dataset = Subset(dataset, test_indices)

    return train_dataset, test_dataset


def create_train_val_test_sets(dataset: TensorDataset):
    """
        Expects data/interim/gc10_tensors.pt to exist.
        Create gc10_train.pt, gc10_val.pt, gc10_test.pt from successive stratified splits.
        # TODO move this method into test_imagedataset.py
    """
    # TODO assert that each class has a minimum number of instances in the test/val sets
    train_val_data, test_data = stratified_split(dataset, train_ratio=0.8, random_seed=15)
    # stratified split returns Subsets, turn train_val_data into a TensorDataset to split it again
    train_data, val_data = stratified_split(TensorDataset(dataset.tensors[0][train_val_data.indices],
                                                          dataset.tensors[1][train_val_data.indices].type(LongTensor)),
                                            random_seed=15,
                                            train_ratio=0.9)
    train_path = project_path('data/interim/gc10_train.pt')
    if not os.path.isfile(train_path):
        torch.save(train_data, train_path)

    val_path = project_path('data/interim/gc10_val.pt')
    if not os.path.isfile(val_path):
        torch.save(val_data, val_path)

    test_path = project_path('data/interim/gc10_val.pt')
    if not os.path.isfile(test_path):
        torch.save(test_data, test_path)

    return train_data, val_data, test_data