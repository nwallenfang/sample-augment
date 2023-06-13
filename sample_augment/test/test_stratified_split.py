from pathlib import Path

import numpy as np
import torch

from sample_augment.data.imagefolder_to_tensors import AugmentDataset
from sample_augment.data.train_test_split import stratified_split


def split_dummy_data(n, min_test_instances, number_of_1s=4):
    dummy_data = torch.ones([n, 1, 1])
    dummy_labels = torch.zeros(n, dtype=torch.long)
    assert number_of_1s < n
    dummy_labels[:number_of_1s + 1] = 1
    # shuffle labels
    dummy_labels = dummy_labels[torch.randperm(dummy_labels.shape[0])]
    dataset = AugmentDataset('dummy', dummy_data, dummy_labels, ['/'] * n, Path('/'))

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


def test_repeated_split():
    n = 1000
    dummy_data = torch.ones([n, 1, 1])
    dummy_labels = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        dummy_data[i] = i

    dataset = AugmentDataset(image_tensor=dummy_data, label_tensor=dummy_labels,
                             img_ids=[Path() for _ in range(n)], root_dir=Path(), name='test')

    train_val, test = stratified_split(dataset, random_seed=42, min_instances_per_class=0)
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
    n = 1000
    dummy_data = torch.ones([n, 1, 1])
    for i in range(n):
        dummy_data[i][:] = i
    dummy_labels = torch.zeros(n, dtype=torch.long)
    dataset = AugmentDataset(image_tensor=dummy_data, label_tensor=dummy_labels,
                             img_ids=[Path() for _ in range(n)], root_dir=Path(), name='test')

    train_1, test_1 = stratified_split(dataset, random_seed=42)
    train_2, test_2 = stratified_split(dataset, random_seed=42)

    some_data_1 = [train_1[i] for i in range(20)]
    some_data_2 = [train_2[i] for i in range(20)]

    some_test_data_1 = [test_1[i] for i in range(20)]
    some_test_data_2 = [test_2[i] for i in range(20)]

    assert some_data_1 == some_data_2
    assert some_test_data_1 == some_test_data_2

    train_1, test_1 = stratified_split(dataset, random_seed=42)
    train_2, test_2 = stratified_split(dataset, random_seed=43)

    some_data_1 = [train_1[i] for i in range(20)]
    some_data_2 = [train_2[i] for i in range(20)]

    some_test_data_1 = [test_1[i] for i in range(20)]
    some_test_data_2 = [test_2[i] for i in range(20)]

    assert some_data_1 != some_data_2
    assert some_test_data_1 != some_test_data_2


def test_stratified_split():
    split_dummy_data(n=20, min_test_instances=3, number_of_1s=4)
    split_dummy_data(n=100, min_test_instances=10, number_of_1s=13)
    # fails and should print an error message
    # run_test(n=100, min_test_instances=10, number_of_1s=5)
