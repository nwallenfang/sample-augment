import glob
from collections import defaultdict
from pathlib import Path
from typing import Optional, Dict, List

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms import ToTensor

from sample_augment.core import Artifact, step
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


class SynthData(Artifact):
    synthetic_images: Tensor
    synthetic_labels: Tensor
    multi_label: bool


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
            temp_tensors.append(image_tensor)
            temp_labels.append(label_tensor)
            augmented_ids.append(f"{generator_name}_{img_path.name.split('.')[0]}")

    augmented_tensors = torch.cat([augmented_tensors] + temp_tensors, 0)
    augmented_labels = torch.cat([augmented_labels] + temp_labels, 0)

    # Convert tensors to uint8
    augmented_tensors = (augmented_tensors * 255).byte()

    return UnifiedTrainSet(name=f"synth-aug-{generator_name}", root_dir=generated_dir, img_ids=augmented_ids,
                           tensors=(augmented_tensors, augmented_labels))


class SynthAugmentedTrain(TrainSet):
    """
        Synthetically Augmented Training Set
    """
    serialize_this = False
    synthetic_images: Tensor
    synthetic_labels: Tensor
    synth_p: float
    """if the synthetic strategy only supports single-label creation, set this to False"""
    multi_label: bool
    _label_to_indices: Optional[Dict] = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # precompute label_to_indices dict to make __getitem__ easier and more performant
        self._label_to_indices = defaultdict(list)
        # could do some dimensionality checks here for the synth_images and labels
        for i, label in enumerate(self.synthetic_labels):
            label_on_cpu = label.cpu() if device == "cuda" else label
            self._label_to_indices[tuple(label_on_cpu.tolist())].append(i)

        if device == "cuda":
            self.synthetic_images = self.synthetic_images.cuda()
            self.synthetic_labels = self.synthetic_labels.cuda()

    def __len__(self):
        return super().__len__()

    def __getitem__(self, idx):
        image, label = super().__getitem__(idx)
        if np.random.rand() < self.synth_p:
            log.warning("DEBUG SYNTH P EVENT")
            if self.multi_label:
                label_key = tuple(label.cpu().tolist())
            else:
                # if we only have single-label synth data, use the primary label
                label_key = [0.0] * 10
                label_key[self.primary_label_tensor[idx].cpu().item()] = 1.0
                label_key = tuple(label_key)

            matching_indices = self._label_to_indices.get(label_key, [])
            if matching_indices:
                chosen_idx = np.random.choice(matching_indices)
                synthetic_image, synthetic_label = (self.synthetic_images[chosen_idx],
                                                    self.synthetic_labels[
                                                        chosen_idx].cpu())
                if self.transform is not None:
                    synthetic_image = self.transform(synthetic_image)
                else:
                    log.warning('Using SynthAugTrainSet without the transform attribute being set.')
                return synthetic_image, synthetic_label
            else:
                log.warning(f'No matching indices for label {label.cpu().tolist()}.')

        # self.transform got applied already in __super__ call
        return image, label

    class Config:  # needed for _label_to_indices to be allowed
        extra = "allow"


class SyntheticBundle(Artifact):
    synthetic_datasets: List[SynthData]
