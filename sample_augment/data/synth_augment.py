import glob
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import ToTensor

from sample_augment.core import step
from sample_augment.data.train_test_split import TrainSet, ValSet
from sample_augment.models.generator import GC10_CLASSES
from sample_augment.models.train_classifier import TrainedClassifier, train_classifier
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir


class AugmentedTrainSet(TrainSet):
    serialize_this = True  # good to have serialized
    # pretty confusing naming, AugmentDataset should be called CompleteDataset
    pass


class SeparatedAugmentedTrainSet(TrainSet):
    synthetic_images: Tensor
    synthetic_labels: Tensor
    synthetic_ids: List[str]


@step
def synth_augment(training_set: TrainSet, generator_name: str = "apa-020") -> AugmentedTrainSet:
    """
        first most basic synthetic augmentation type.
        Gets a directory of generated images and adds those to the smaller classes, until each class
        has at least the average instance count from before
    """
    # it's doubtful to say the least that StyleGAN will be able to learn the class with 8 (or even 11) instances.
    # possibly we'll have to resort to putting the validation instances into for the small classes into the train set
    # for StyleGAN and accept that our F1 estimation will be a little optimistic for these classes
    class_counts = np.bincount(training_set.tensors[1].numpy())
    _median_count = int(np.median(class_counts) + 0.5)
    target_count = 200  # = median_count
    augmented_tensors = training_set.image_tensor.detach().clone()
    augmented_labels = training_set.label_tensor.detach().clone()
    augmented_ids = training_set.img_ids.copy()

    generated_dir = shared_dir / "generated" / generator_name
    temp_tensors = []
    temp_labels = []
    # stock up each class up to median count
    for class_idx, class_name in enumerate(GC10_CLASSES):
        n_missing = target_count - class_counts[class_idx]
        if n_missing <= 0:
            log.info(f"Augment: skip {class_name}")
            continue

        # Find the images for the class_name
        generated_image_paths = glob.glob(
            f"{generated_dir}/{class_name}_*.png")  # This will return a list of all files that match the pattern

        # Use min to avoid trying to add more images than exist
        for j in range(min(n_missing, len(generated_image_paths))):
            img_path = Path(generated_image_paths[j])
            image = Image.open(img_path)
            image_tensor = ToTensor()(image).unsqueeze(0)  # Add batch dimension

            # Create a tensor for the label and add a dimension
            label_tensor = torch.tensor([class_idx], dtype=torch.long)  # Assuming that `i` is your label

            # Append the image_tensor and label_tensor to augmented_tensors and augmented_labels
            # todo check if I need to do any preprocessing
            temp_tensors.append(image_tensor)
            temp_labels.append(label_tensor)
            augmented_ids.append(f"{generator_name}_{img_path.name.split('.')[0]}")

    augmented_tensors = torch.cat([augmented_tensors] + temp_tensors, 0)
    augmented_labels = torch.cat([augmented_labels] + temp_labels, 0)

    # Convert tensors to uint8
    augmented_tensors = (augmented_tensors * 255).byte()

    return AugmentedTrainSet(name=f"synth-aug-{generator_name}", root_dir=generated_dir, img_ids=augmented_ids,
                             tensors=(augmented_tensors, augmented_labels))


@step
def synth_augment_seperate(training_set: TrainSet, generator_name: str) -> SeparatedAugmentedTrainSet:
    """
        first most basic synthetic augmentation type.
        Gets a directory of generated images and adds those to the smaller classes, until each class
        has at least the average instance count from before
    """
    # it's doubtful to say the least that StyleGAN will be able to learn the class with 8 (or even 11) instances.
    # possibly we'll have to resort to putting the validation instances into for the small classes into the train set
    # for StyleGAN and accept that our F1 estimation will be a little optimistic for these classes
    class_counts = np.bincount(training_set.tensors[1].numpy())
    _median_count = int(np.median(class_counts) + 0.5)
    target_count = 200  # = median_count
    augmented_tensors = training_set.image_tensor.detach().clone()
    augmented_labels = training_set.label_tensor.detach().clone()
    augmented_ids = training_set.img_ids.copy()

    generated_dir = shared_dir / "generated" / generator_name
    temp_tensors = []
    temp_labels = []
    # stock up each class up to median count
    for class_idx, class_name in enumerate(GC10_CLASSES):
        n_missing = target_count - class_counts[class_idx]
        if n_missing <= 0:
            log.info(f"Augment: skip {class_name}")
            continue

        # Find the images for the class_name
        generated_image_paths = glob.glob(
            f"{generated_dir}/{class_name}_*.png")  # This will return a list of all files that match the pattern

        # Use min to avoid trying to add more images than exist
        for j in range(min(n_missing, len(generated_image_paths))):
            img_path = Path(generated_image_paths[j])
            image = Image.open(img_path)
            image_tensor = ToTensor()(image).unsqueeze(0)  # Add batch dimension

            # Create a tensor for the label and add a dimension
            label_tensor = torch.tensor([class_idx], dtype=torch.long)  # Assuming that `i` is your label

            # Append the image_tensor and label_tensor to augmented_tensors and augmented_labels
            # todo check if I need to do any preprocessing
            temp_tensors.append(image_tensor)
            temp_labels.append(label_tensor)
            augmented_ids.append(f"{generator_name}_{img_path.name.split('.')[0]}")

    augmented_tensors = torch.cat([augmented_tensors] + temp_tensors, 0)
    augmented_labels = torch.cat([augmented_labels] + temp_labels, 0)

    # Convert tensors to uint8
    augmented_tensors = (augmented_tensors * 255).byte()

    return AugmentedTrainSet(name=f"synth-aug-{generator_name}", root_dir=generated_dir, img_ids=augmented_ids,
                             tensors=(augmented_tensors, augmented_labels))


@step
def train_augmented_classifier(train_data: AugmentedTrainSet, val_data: ValSet,
                               num_epochs: int, batch_size: int, learning_rate: float,
                               balance_classes: bool,
                               random_seed: int,
                               data_augment: bool,
                               geometric_augment: bool,
                               color_jitter: float,
                               h_flip_p: float,
                               v_flip_p: float) -> TrainedClassifier:
    return train_classifier(train_data, val_data, num_epochs, batch_size, learning_rate, balance_classes, random_seed,
                            data_augment, geometric_augment, color_jitter, h_flip_p, v_flip_p)


@step
def look_at_augmented_train_set(augmented: AugmentedTrainSet):
    print(augmented.name)
