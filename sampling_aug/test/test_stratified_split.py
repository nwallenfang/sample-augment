import numpy as np
import torch
from torch.utils.data import TensorDataset

from data.train_test_split import stratified_split


def run_test(n, min_test_instances, number_of_1s=4):
    dummy_data = torch.ones([n, 1, 1])
    dummy_labels = torch.zeros(n, dtype=torch.long)
    assert number_of_1s < n
    dummy_labels[:number_of_1s + 1] = 1
    # shuffle labels
    dummy_labels = dummy_labels[torch.randperm(dummy_labels.shape[0])]
    dataset = TensorDataset(dummy_data, dummy_labels)

    train, test = stratified_split(dataset, random_seed=42, min_instances_per_class=min_test_instances)

    test_labels = dataset.tensors[1][test.indices].numpy()
    train_labels = dataset.tensors[1][train.indices].numpy()

    test_counts = np.bincount(test_labels)
    train_counts = np.bincount(train_labels)

    assert len(dataset) == n
    assert len(test_labels) + len(train_labels) == n
    assert len(train.indices) + len(test.indices) == n
    assert test_counts[1] >= min_test_instances
    assert train_counts[1] <= min_test_instances


def test_stratified_split():
    run_test(n=20, min_test_instances=3, number_of_1s=4)
    run_test(n=100, min_test_instances=10, number_of_1s=13)
    # fails and should print an error message
    # run_test(n=100, min_test_instances=10, number_of_1s=5)
