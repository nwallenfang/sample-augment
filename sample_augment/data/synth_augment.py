import numpy as np

from sample_augment.core import step
from sample_augment.data.train_test_split import TrainSet, TestSet, ValSet
from sample_augment.models.generator import GC10_CLASSES


class AugmentedDataset(TrainSet):
    _serialize_this = True  # good to have serialized
    # TODO pretty confusing naming, AugmentDataset should be called CompleteDataset
    pass


@step
def synth_augment(training_set: TrainSet, val_set: ValSet, test_set: TestSet):
    """
        first most basic synthetic augmentation type.
        Gets a directory of generated images and adds those to the smaller classes, until each class
        has at least the average instance count from before
    """
    class_counts = np.bincount(training_set.tensors[1].numpy())
    val_counts = np.bincount(val_set.tensors[1].numpy())
    test_counts = np.bincount(test_set.tensors[1].numpy())

    # it's doubtful to say the least that StyleGAN will be able to learn the class with 8 (or even 11) instances.
    # possibly we'll have to resort to putting the validation instances into for the small classes into the train set
    # for StyleGAN and accept that our F1 estimation will be a little optimistic for these classes
    for i, class_name in enumerate(GC10_CLASSES):
        print(f"{class_name}: {class_counts[i]} training instances.")
        print(f"{class_name}: {val_counts[i]} validation instances.")
        print(f"{class_name}: {test_counts[i]} test instances.")
