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
from sample_augment.models.generator import GC10_CLASSES, StyleGANGenerator
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir
from sample_augment.utils.plot import show_image_tensor


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

    # TODO give mapping attribute
    # precompute the label_to_indices mapping
    # self.label_to_indices = defaultdict(list)
    # for i, label in enumerate(self.synthetic_labels):
    #     # Assuming label is a tensor, converting it to a tuple to use it as a key
    #     self.label_to_indices[tuple(label.tolist())].append(i)

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        # the super __getitem__ already has self.transform applied
        image, label = super().__getitem__(idx)

        if np.random.rand() < self.synth_p:
            # FIXME something is very imperformant here
            #  maybe we do need an additional data structure matching label to indices
            #  and / or the synthetic data should be on GPU
            # TODO
            #     matching_indices = self.label_to_indices.get(tuple(label.tolist()), [])
            matching_rows = (self.synthetic_labels.bool() == label.bool()).all(dim=1)
            matching_indices = torch.where(matching_rows)[0]

            if len(matching_indices) > 0:  # If there are synthetic instances with the same label
                # Replace the real instance with a synthetic one

                chosen_idx = np.random.choice(matching_indices.cpu().numpy())
                synthetic_image, synthetic_label = (self.synthetic_images[chosen_idx],
                                                    self.synthetic_labels[chosen_idx])
                if self.transform is not None:
                    synthetic_image = self.transform(synthetic_image)
                else:
                    log.warning('Using TrainSetWithSynthetic without the transform being set.')
                return synthetic_image, synthetic_label

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
            # TODO this skipping doesn't make anymore sense now with synth_p, no?
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
def synth_augment_online(training_set: TrainSet, generator_name: str, synth_p: float) -> TrainSetWithSynthetic:
    """
    A synthetic augmentation type based on label distribution.
    """

    label_matrix = training_set.label_tensor.numpy()
    unique_label_combinations = np.unique(label_matrix, axis=0)

    # around 50
    log.info(f"Number of unique label combinations: {len(unique_label_combinations)}")
    n = 10  # generate 10 synthetic instances per label combination
    n_combinations = len(unique_label_combinations)

    synthetic_imgs_np = np.empty(shape=(n * n_combinations, 256, 256, 3), dtype=np.uint8)
    synthetic_labels = []
    synthetic_ids = []

    generator = StyleGANGenerator.load_from_name(generator_name)

    for label_idx, label_comb in enumerate(unique_label_combinations):
        # stack label_comb vector n times to generate n images
        c = np.tile(label_comb, (n, 1))  # shape will be [n, num_classes]
        synthetic_instances = generator.generate(c=c)
        synthetic_imgs_np[label_idx * n:(label_idx + 1) * n] = synthetic_instances
        synthetic_labels.extend([label_comb for _ in range(n)])
        synthetic_ids.extend([f"{generator_name}_{label_comb}_{i}" for i in range(n)])

    synthetic_imgs_tensor = torch.from_numpy(synthetic_imgs_np)
    synthetic_labels = torch.from_numpy(np.array(synthetic_labels))
    log.info(f"Label size: {synthetic_labels.size()}")

    # Convert tensors to uint8
    # synthetic_imgs_tensor = (synthetic_imgs_tensor * 255).byte()

    # permute synthetic tensor to be (N, C, H, W), same as training data
    synthetic_imgs_tensor = synthetic_imgs_tensor.permute(0, 3, 1, 2)

    return TrainSetWithSynthetic(name=f"synth-aug-{generator_name}", root_dir=training_set.root_dir,
                                 img_ids=training_set.img_ids,
                                 tensors=(training_set.tensors[0], training_set.tensors[1]),
                                 primary_label_tensor=training_set.primary_label_tensor,
                                 synthetic_images=synthetic_imgs_tensor,
                                 synthetic_labels=synthetic_labels,
                                 synthetic_ids=synthetic_ids,
                                 synth_p=synth_p)


@step
def look_at_augmented_train_set(augmented: TrainSetWithSynthetic):
    for i in range(50):
        _img = augmented[i]

    for label_combo in range(19, 24):
        for i in range(10 * label_combo, 10 * (label_combo + 1)):
            img_arr = augmented.synthetic_images[i]
            labels = augmented.synthetic_labels[i]
            class_names = [class_name for class_name, label in zip(GC10_CLASSES, np.array(labels)) if label == 1.0]
            show_image_tensor(img_arr, title=str(class_names))
