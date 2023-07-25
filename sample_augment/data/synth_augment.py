import glob
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import ToTensor

from sample_augment.core import step
from sample_augment.data.train_test_split import TrainSet
from sample_augment.models.generator import GC10_CLASSES
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir


class UnifiedTrainSet(TrainSet):
    """
        A training set where synthetic images have been added.
    """
    serialize_this = True  # good to have serialized
    pass


@step
def synth_augment_unified(training_set: TrainSet, generator_name: str = "apa-020") -> UnifiedTrainSet:
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

    return UnifiedTrainSet(name=f"synth-aug-{generator_name}", root_dir=generated_dir, img_ids=augmented_ids,
                           tensors=(augmented_tensors, augmented_labels))


class TrainSetWithSynthetic(TrainSet):
    serialize_this = True
    # along with the normal TrainSet tensors:
    synthetic_images: Tensor
    synthetic_labels: Tensor
    synthetic_ids: List[str]
    """probability of replacing a real training image with a synthetic one"""
    synth_p: float

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        primary_label = self.primary_label_tensor[idx]
        if np.random.rand() < self.synth_p:
            same_class_indices = (self.synthetic_labels == primary_label).nonzero(as_tuple=True)[0]
            if len(same_class_indices) > 0:  # If there are synthetic instances from the same class
                # Replace the real instance with a synthetic one
                chosen_idx = np.random.choice(same_class_indices)
                synthetic_image, synthetic_label = (self.synthetic_images[chosen_idx],
                                                    self.synthetic_labels[chosen_idx])
                # TODO should we run data augmentation on synthetic images as well?
                #  if not need two different transform objects
                synthetic_image = self.transform(synthetic_image)
                synth_label_one_hot = torch.zeros_like(label)
                synth_label_one_hot[primary_label] = 1.0
                log.info(f"synthetic {self.synthetic_ids[chosen_idx.item()]}")
                return synthetic_image, synth_label_one_hot

        return image, label


@step
def synth_augment(training_set: TrainSet, generator_name: str, synth_p: float) -> TrainSetWithSynthetic:
    """
    First most basic synthetic augmentation type.
    Gets a directory of generated images and adds those to the smaller classes, until each class
    has at least the average instance count from before.
    """
    class_counts = np.bincount(training_set.primary_label_tensor.numpy())
    _median_count = int(np.median(class_counts) + 0.5)
    target_count = 200  # = median_count

    generated_dir = shared_dir / "generated" / generator_name

    # Store tensors and labels for synthetic data
    synthetic_tensors = []
    synthetic_labels = []
    synthetic_ids = []

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

            # Append the image_tensor and label_tensor to synthetic_tensors and synthetic_labels
            # todo check if I need to do any preprocessing
            synthetic_tensors.append(image_tensor)
            synthetic_labels.append(label_tensor)
            synthetic_ids.append(f"{generator_name}_{img_path.name.split('.')[0]}")

    synthetic_tensors = torch.cat(synthetic_tensors, 0)
    synthetic_labels = torch.cat(synthetic_labels, 0)

    # Convert tensors to uint8
    synthetic_tensors = (synthetic_tensors * 255).byte()

    return TrainSetWithSynthetic(name=f"synth-aug-{generator_name}", root_dir=generated_dir,
                                 img_ids=training_set.img_ids,
                                 tensors=(training_set.tensors[0], training_set.tensors[1]),
                                 primary_label_tensor=training_set.primary_label_tensor,
                                 synthetic_images=synthetic_tensors,
                                 synthetic_labels=synthetic_labels,
                                 synthetic_ids=synthetic_ids,
                                 synth_p=synth_p)


@step
def look_at_augmented_train_set(augmented: UnifiedTrainSet):
    print(augmented.name)
