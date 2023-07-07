"""
    Perform a stratified train test split (balance classes).

"""
import typing

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

from sample_augment.core import step, Artifact
from sample_augment.data.dataset import AugmentDataset
from sample_augment.utils.log import log


def stratified_split(dataset: AugmentDataset,
                     train_ratio: float = 0.8,
                     random_seed: int = 42,
                     min_instances_per_class: int = 10):
    """
        Perform a random stratified split of a Dataset into two Datasets (called train and test set).
    Args:
        dataset(AugmentDataset): Dataset instance to split,
            can be an instance of type CustomTensorDataset.
        train_ratio: Ratio of training set size in relation to total size
        random_seed:
        min_instances_per_class: Minimum number of instances of each class that will be in the test set
    """
    np.random.seed(random_seed)

    labels = dataset.label_tensor

    n = len(dataset)
    indices = list(range(n))

    # use scikit-learn stratified split on the indices
    train_indices, test_indices = train_test_split(indices, train_size=train_ratio,
                                                   stratify=labels, random_state=random_seed)
    train_indices, test_indices = np.array(train_indices), np.array(test_indices)
    # calculate instance counts per class of the test set,
    # shift instances from the training set to ensure min count
    class_counts = np.bincount(labels[test_indices])
    for class_no in np.where(class_counts < min_instances_per_class)[0]:
        class_no = class_no.item()
        number_of_instances = min_instances_per_class - class_counts[class_no]
        # randomly choose n_missing instances from train to put into test set
        # class_indices is an array of indices into train_indices,
        # which all have the label of the needed class
        class_indices = np.argwhere(labels[train_indices] == class_no).flatten()
        if len(class_indices) < number_of_instances:
            log.error(f'stratified split: Not enough instances of class with label {class_no} to fulfill '
                      f'min_instances={min_instances_per_class}')
            continue
        # choose the indices that will be moved from train to test randomly
        indices_to_move = np.random.choice(class_indices, number_of_instances, replace=False)
        # actual indices of the instances (labels tensor) that will be moved
        # we're working on indices of indices (since it's a Subset), which can be confusing!
        instances_to_move = train_indices[indices_to_move]

        test_indices = np.append(test_indices, instances_to_move)
        train_indices = np.delete(train_indices, indices_to_move)

    assert set(train_indices).intersection(set(test_indices)) == set()

    # take metadata (img paths) from sample_augment.dataset into subsets
    train_dataset = dataset.subset(train_indices, name="train")
    test_dataset = dataset.subset(test_indices, name="test")

    return train_dataset, test_dataset


class FoldDatasets(Artifact):
    _serialize_this = False
    datasets: typing.List[typing.Tuple[AugmentDataset, AugmentDataset]]


@step
def stratified_k_fold_split(dataset: AugmentDataset, n_folds: int = 5, random_seed: int = 42,
                            min_instances_per_class: int = 10) -> FoldDatasets:
    np.random.seed(random_seed)
    labels = dataset.label_tensor
    n = len(dataset)
    indices = list(range(n))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    fold_datasets = []
    for train_indices, test_indices in skf.split(indices, labels):
        train_indices, test_indices = np.array(train_indices), np.array(test_indices)
        class_counts = np.bincount(labels[test_indices])
        for class_no in np.where(class_counts < min_instances_per_class)[0]:
            class_no = class_no.item()
            number_of_instances = min_instances_per_class - class_counts[class_no]
            class_indices = np.argwhere(labels[train_indices] == class_no).flatten()
            if len(class_indices) < number_of_instances:
                log.error(
                    f'stratified split: Not enough instances of class with label {class_no} '
                    f'to fulfill min_instances={min_instances_per_class}')
                continue
            indices_to_move = np.random.choice(class_indices, number_of_instances, replace=False)
            instances_to_move = train_indices[indices_to_move]
            test_indices = np.append(test_indices, instances_to_move)
            train_indices = np.delete(train_indices, indices_to_move)

        assert set(train_indices).intersection(set(test_indices)) == set()
        train_dataset = dataset.subset(train_indices, name=f"train_fold_{len(fold_datasets) + 1}")
        test_dataset = dataset.subset(test_indices, name=f"test_fold_{len(fold_datasets) + 1}")
        fold_datasets.append((train_dataset, test_dataset))

    return FoldDatasets(datasets=fold_datasets)


class TrainSet(AugmentDataset):
    _serialize_this = False
    transform: typing.Optional[typing.Callable] = None

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index].long()

        return x, y


class ValSet(AugmentDataset):
    _serialize_this = False
    transform: typing.Optional[typing.Callable] = None

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index].long()

        return x, y


class TestSet(AugmentDataset):
    _serialize_this = False


class TrainTestValBundle(Artifact):
    train: TrainSet
    val: ValSet
    test: TestSet
    _serialize_this = False


# helper classes and steps to have clean interfaces
# since currently the steps API only supports returning a single Artifact


@step
def extract_train_set(bundle: TrainTestValBundle) -> TrainSet:
    train_set = typing.cast(TrainSet, bundle.train)
    return train_set


@step
def extract_val_set(bundle: TrainTestValBundle) -> ValSet:
    return typing.cast(ValSet, bundle.val)


@step
def extract_test_set(bundle: TrainTestValBundle) -> TestSet:
    return typing.cast(TestSet, bundle.test)


@step
def create_train_test_val(dataset: AugmentDataset, random_seed: int,
                          test_ratio: float, val_ratio: float, min_instances: int) -> TrainTestValBundle:
    n = len(dataset)
    train_val_data, test_data = stratified_split(dataset,
                                                 train_ratio=(1.0 - test_ratio),
                                                 random_seed=random_seed,
                                                 min_instances_per_class=min_instances)
    # stratified split returns Subsets, turn train_val_data into a TensorDataset to split it again
    train_data, val_data = stratified_split(train_val_data,
                                            random_seed=random_seed,
                                            train_ratio=(1.0 - val_ratio),
                                            min_instances_per_class=min_instances)

    assert len(train_data) + len(val_data) == len(train_val_data)
    assert len(train_val_data) + len(test_data) == n
    trainset, valset, testset = TrainSet.from_existing(train_data), ValSet.from_existing(val_data), \
        TestSet.from_existing(test_data)
    assert len(trainset) + len(valset) == len(train_val_data)
    assert len(train_val_data) + len(testset) == n
    return TrainTestValBundle(train=trainset, val=valset, test=testset)
