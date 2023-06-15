from pathlib import Path

import numpy as np
import pytest
import torch

from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.train_test_split import stratified_split


def create_dummy_dataset(n, number_of_1s, root_dir=None):
    dummy_data = torch.ones([n, 1, 1])

    for i in range(n):
        dummy_data[i] = i

    dummy_labels = torch.zeros(n, dtype=torch.long)
    assert number_of_1s < n
    dummy_labels[:number_of_1s + 1] = 1
    # shuffle labels
    dummy_labels = dummy_labels[torch.randperm(dummy_labels.shape[0])]
    if root_dir:
        dataset = AugmentDataset(name='dummy', tensors=(dummy_data, dummy_labels),
                                 img_ids=[str(i) for i in range(n)],
                                 root_dir=root_dir)
    else:
        dataset = AugmentDataset(name='dummy', tensors=(dummy_data, dummy_labels),
                                 img_ids=[str(i) for i in range(n)],
                                 root_dir=Path('/'))
    assert len(dataset) == n
    return dataset


@pytest.mark.parametrize("n, number_of_1s, min_test_instances", [
    (20, 4, 3),
    (100, 13, 10),
])
def test_split_dummy_data(min_test_instances, n, number_of_1s):
    dummy_dataset = create_dummy_dataset(n=n, number_of_1s=number_of_1s)
    train, test = stratified_split(dummy_dataset, random_seed=42, min_instances_per_class=min_test_instances)

    test_labels = test.label_tensor.numpy()
    train_labels = train.label_tensor.numpy()

    test_counts = np.bincount(test_labels)
    train_counts = np.bincount(train_labels)

    assert len(test_labels) + len(train_labels) == n
    assert test_counts[1] >= min_test_instances
    assert train_counts[1] <= min_test_instances


def test_repeated_split(n=100):
    dummy_dataset = create_dummy_dataset(n, n // 5)
    train_val, test = stratified_split(dummy_dataset, random_seed=42, min_instances_per_class=0)
    train, val = stratified_split(train_val, random_seed=42, min_instances_per_class=0)

    # check that every label can be found in either train, val, or test
    # and check that there are no duplicates
    # (if this fails for the repeated split, test it for the single split first)
    train_data = train.tensors[0].flatten().tolist()
    test_data = test.tensors[0].flatten().tolist()
    val_data = val.tensors[0].flatten().tolist()

    for i in range(n):
        if i in train_data:
            assert (i not in test_data and i not in val_data)
        elif i in test_data:
            assert (i not in train_data and i not in val_data)
        elif i in val_data:
            assert (i not in train_data and i not in test_data)
        else:  # has to be in one of those three sets
            print(i, "in neither test, train, val sets")
            assert False
        # assert ((i in train_labels and i not in test_labels and i not in val_labels) or
        #         (i not in train_labels and i in test_labels and i not in val_labels) or
        #         (i not in train_labels and i not in test_labels and i in val_labels))
    assert len(train) + len(val) + len(test) == n


def test_split_determinism():
    dummy_dataset = create_dummy_dataset(n=1000, number_of_1s=400)
    train_1, test_1 = stratified_split(dummy_dataset, random_seed=42)
    train_2, test_2 = stratified_split(dummy_dataset, random_seed=42)

    some_data_1 = [train_1[i] for i in range(20)]
    some_data_2 = [train_2[i] for i in range(20)]

    some_test_data_1 = [test_1[i] for i in range(20)]
    some_test_data_2 = [test_2[i] for i in range(20)]

    assert some_data_1 == some_data_2
    assert some_test_data_1 == some_test_data_2

    train_1, test_1 = stratified_split(dummy_dataset, random_seed=42)
    train_2, test_2 = stratified_split(dummy_dataset, random_seed=43)

    some_data_1 = [train_1[i] for i in range(20)]
    some_data_2 = [train_2[i] for i in range(20)]

    some_test_data_1 = [test_1[i] for i in range(20)]
    some_test_data_2 = [test_2[i] for i in range(20)]

    assert some_data_1 != some_data_2
    assert some_test_data_1 != some_test_data_2
