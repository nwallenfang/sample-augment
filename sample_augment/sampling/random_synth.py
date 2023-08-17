import torch

from sample_augment.data.train_test_split import TrainSet
from sample_augment.models.generator import StyleGANGenerator
from sample_augment.sampling.synth_augment import SynthData
from sample_augment.utils import log


def random_synthetic_augmentation(training_set: TrainSet, generator_name: str) -> SynthData:
    """
    A synthetic augmentation type based on label distribution.
    """
    # since we're using the generator we're using device == 'cuda' here
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device("cpu")

    label_matrix = training_set.label_tensor.to(device)
    unique_label_combinations, _ = torch.unique(label_matrix, dim=0, return_inverse=True)

    log.info(f"Number of unique label combinations: {len(unique_label_combinations)}")
    n = 10  # generate 10 synthetic instances per label combination
    n_combinations = len(unique_label_combinations)

    # Initialize tensors to hold synthetic images and labels
    synthetic_imgs_tensor = torch.empty((n * n_combinations, 3, 256, 256), dtype=torch.uint8,
                                        device=device)
    synthetic_labels_tensor = torch.empty((n * n_combinations, label_matrix.size(1)), device=device,
                                          dtype=torch.float32)

    generator = StyleGANGenerator.load_from_name(generator_name)

    for label_idx, label_comb in enumerate(unique_label_combinations):
        c = label_comb.repeat(n, 1)
        synthetic_instances = generator.generate(c=c)
        synthetic_imgs_tensor[label_idx * n:(label_idx + 1) * n] = synthetic_instances.permute(0, 3, 1, 2)
        synthetic_labels_tensor[label_idx * n:(label_idx + 1) * n] = c

    _synthetic_ids = [f"{generator_name}_{label_comb.tolist()}_{i}" for label_comb in unique_label_combinations for i in
                      range(n)]

    # Move everything to CPU before returning
    synthetic_imgs_tensor = synthetic_imgs_tensor.cpu()
    synthetic_labels_tensor = synthetic_labels_tensor.cpu()

    return SynthData(synthetic_images=synthetic_imgs_tensor, synthetic_labels=synthetic_labels_tensor,
                     multi_label=True)
