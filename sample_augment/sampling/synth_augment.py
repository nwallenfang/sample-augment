import glob
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import ToTensor

from sample_augment.core import step
from sample_augment.data.synth_data import SynthAugmentedTrain
from sample_augment.data.train_test_split import TrainSet
from sample_augment.models.generator import GC10_CLASSES, StyleGANGenerator
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir
from sample_augment.utils.plot import show_image_tensor


# TODO restructure the steps here in this file so they receive SynthData


@step
def synth_augment(training_set: TrainSet, generator_name: str, synth_p: float) -> SynthAugmentedTrain:
    """
    First most basic synthetic augmentation type.
    Gets a directory of generated images and adds those to the smaller classes, until each class
    has at least the average instance count from before.
    """
    class_counts = np.bincount(training_set.primary_label_tensor.numpy())
    _median_count = int(np.median(class_counts) + 0.5)
    _target_count = 200  # = median_count

    generated_dir = shared_dir / "generated" / generator_name

    log.info(f'synth-augment generated dir: {generated_dir}')

    # Store tensors and labels for synthetic data
    synthetic_tensors = []
    synthetic_labels = []
    synthetic_ids = []

    for class_idx, class_name in enumerate(GC10_CLASSES):
        _n_missing = _target_count - class_counts[class_idx]

        # Find the images for the class_name
        generated_image_paths = glob.glob(
            f"{generated_dir}/{class_name}/*.jpg")
        n_imgs = len(generated_image_paths)
        # Use min to avoid trying to add more images than exist
        # n_adding = min(n_missing, len(generated_image_paths))
        # Just add every img for our use-case now
        log.info(f'Augmenting {n_imgs} {class_name} instances.')
        for j in range(n_imgs):
            img_path = Path(generated_image_paths[j])
            image = Image.open(img_path)
            image_tensor = ToTensor()(image).unsqueeze(0)  # Add batch dimension

            # Create a tensor for the label and add a dimension
            label_tensor = torch.zeros(10, dtype=torch.float32)
            label_tensor[class_idx] = 1.0

            # Append the image_tensor and label_tensor to synthetic_tensors and synthetic_labels
            # todo check if I need to do any preprocessing
            synthetic_tensors.append(image_tensor)
            synthetic_labels.append(label_tensor)
            synthetic_ids.append(f"{generator_name}_{img_path.name.split('.')[0]}")

    synthetic_tensors = torch.cat(synthetic_tensors, 0)
    # synthetic
    synthetic_labels = torch.stack(synthetic_labels, 0)

    # Convert tensors to uint8
    synthetic_tensors = (synthetic_tensors * 255).byte()

    return SynthAugmentedTrain(name=f"synth-aug-{generator_name}", root_dir=generated_dir,
                               img_ids=training_set.img_ids,
                               tensors=(training_set.tensors[0], training_set.tensors[1]),
                               primary_label_tensor=training_set.primary_label_tensor,
                               synthetic_images=synthetic_tensors,
                               synthetic_labels=synthetic_labels,
                               synthetic_ids=synthetic_ids,
                               synth_p=synth_p,
                               multi_label=False)  # Could be True depending....


@step
def synth_augment_online(training_set: TrainSet, generator_name: str, synth_p: float,
                         random_seed: int) -> SynthAugmentedTrain:
    """
    A synthetic augmentation type based on label distribution.
    """
    # since we're using the generator we're using device == 'cuda' here
    log.warning("don't use this, see compare_strategies.py")
    device = torch.device('cuda')

    label_matrix = training_set.label_tensor.to(device)
    unique_label_combinations, _ = torch.unique(label_matrix, dim=0, return_inverse=True)

    log.info(f"Number of unique label combinations: {len(unique_label_combinations)}")
    n = 10  # generate 10 synthetic instances per label combination
    n_combinations = len(unique_label_combinations)

    # Initialize tensors to hold synthetic images and labels
    synthetic_imgs_tensor = torch.empty((n * n_combinations, 3, 256, 256), dtype=torch.uint8,
                                        device=device)
    synthetic_labels_tensor = torch.empty((n * n_combinations, label_matrix.size(1)), device=device)

    generator = StyleGANGenerator.load_from_name(generator_name, seed=random_seed)

    for label_idx, label_comb in enumerate(unique_label_combinations):
        c = label_comb.repeat(n, 1)
        synthetic_instances = generator.generate(c=c)
        synthetic_imgs_tensor[label_idx * n:(label_idx + 1) * n] = synthetic_instances.permute(0, 3, 1, 2)
        synthetic_labels_tensor[label_idx * n:(label_idx + 1) * n] = c

    synthetic_ids = [f"{generator_name}_{label_comb.tolist()}_{i}" for label_comb in unique_label_combinations for i in
                     range(n)]

    # Move everything to CPU before returning
    synthetic_imgs_tensor = synthetic_imgs_tensor.cpu()
    synthetic_labels_tensor = synthetic_labels_tensor.cpu()

    return SynthAugmentedTrain(name=f"synth-aug-{generator_name}", root_dir=training_set.root_dir,
                               img_ids=training_set.img_ids,
                               tensors=(training_set.tensors[0], training_set.tensors[1]),
                               primary_label_tensor=training_set.primary_label_tensor,
                               synthetic_images=synthetic_imgs_tensor,
                               synthetic_labels=synthetic_labels_tensor,
                               synthetic_ids=synthetic_ids,
                               synth_p=synth_p)


@step
def look_at_augmented_train_set(augmented: SynthAugmentedTrain):
    for i in range(50):
        _img = augmented[i]

    for label_combo in range(19, 24):
        for i in range(10 * label_combo, 10 * (label_combo + 1)):
            img_arr = augmented.synthetic_images[i]
            labels = augmented.synthetic_labels[i]
            class_names = [class_name for class_name, label in zip(GC10_CLASSES, np.array(labels)) if label == 1.0]
            show_image_tensor(img_arr, title=str(class_names))
