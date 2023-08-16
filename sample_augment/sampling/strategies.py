from typing import List

import sys
import torch
from torch import Tensor

from sample_augment.core import Artifact, step
from sample_augment.data.train_test_split import TrainSet, ValSet
from sample_augment.models.generator import StyleGANGenerator
from sample_augment.models.train_classifier import train_augmented_classifier, TrainedClassifier, train_classifier
from sample_augment.sampling.synth_augment import SynthAugTrainSet
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir


class SynthData(Artifact):
    synthetic_images: Tensor
    synthetic_labels: Tensor
    multi_label: bool


@step
def synth_data_to_training_set(training_set: TrainSet, synth_data: SynthData, generator_name: str, synth_p: float):
    generated_dir = shared_dir / "generated" / generator_name

    return SynthAugTrainSet(name=f"synth-aug-{generator_name}", root_dir=generated_dir,
                            img_ids=training_set.img_ids,
                            tensors=(training_set.tensors[0], training_set.tensors[1]),
                            primary_label_tensor=training_set.primary_label_tensor,
                            synthetic_images=synth_data.synthetic_images,
                            synthetic_labels=synth_data.synthetic_labels,
                            synth_p=synth_p,
                            multi_label=synth_data.multi_label)


def random_synthetic_augmentation(training_set: TrainSet, generator_name: str) -> SynthData:
    """
    A synthetic augmentation type based on label distribution.
    """
    # since we're using the generator we're using device == 'cuda' here
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

    generator = StyleGANGenerator.load_from_name(generator_name)

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
    
    return SynthData(synthetic_images=synthetic_imgs_tensor, synthetic_labels=synthetic_labels_tensor,
                     multi_label=True)
    
def hand_picked_dataset(training_set: TrainSet, generator_name: str):
    generator_name = 'handpicked-' + generator_name
    from sample_augment.sampling.synth_augment import synth_augment
    # a little redundant to do it like that but fine
    synth_aug = synth_augment(training_set=training_set, generator_name=generator_name, synth_p=0.0)
    
    return SynthData(synthetic_images=synth_aug.synthetic_images, synthetic_labels=synth_aug.synthetic_labels,
                     multi_label=False)
    


class SyntheticBundle(Artifact):
    synthetic_datasets: List[SynthData]


ALL_STRATEGIES = {
    "random": random_synthetic_augmentation,
    "hand-picked": hand_picked_dataset
}


@step
def create_synthetic_bundle_from_strategies(strategies: List[str], training_set: TrainSet,
                                            generator_name: str) -> SyntheticBundle:
    synthetic_datasets = []
    for strategy in strategies:
        if strategy not in ALL_STRATEGIES:
            log.error(f'unknown strategy {strategy} provided.')
            sys.exit(-1)
            continue
        strategy_func = ALL_STRATEGIES[strategy]
        synth_set = strategy_func(training_set, generator_name)
        log.info(f'Created set for {strategy}')
        synthetic_datasets.append(synth_set)

    return SyntheticBundle(synthetic_datasets=synthetic_datasets)


class TrainedClassifiersOnSynthData(Artifact):
    baseline: TrainedClassifier
    classifiers: List[TrainedClassifier]

from sample_augment.models.train_classifier import ModelType
@step
def synth_bundle_compare_classifiers(bundle: SyntheticBundle,
                                     strategies: List[str],
                                     train_set: TrainSet,
                                     val_set: ValSet,
                                     synth_p: float,
                                     num_epochs: int, batch_size: int, learning_rate: float,
                                     balance_classes: bool,
                                     model_type: ModelType,
                                     random_seed: int,
                                     data_augment: bool,
                                     geometric_augment: bool,
                                     color_jitter: float,
                                     h_flip_p: float,
                                     v_flip_p: float,
                                     ) -> TrainedClassifiersOnSynthData:
    trained_classifiers: List[TrainedClassifier] = []
    # TODO also train one classifier witout synthetic data
    assert len(bundle.synthetic_datasets) == len(strategies), f'{len(bundle.synthetic_datasets)} datasets != {len(strategies)} strategies'
    for i, synthetic_dataset in enumerate(bundle.synthetic_datasets):
        log.info(f'Training classifier with strategy {strategies[i]}.')
        # this assumes, that all datasets in the bundle used the same generator
        synth_training_set = synth_data_to_training_set(train_set, synthetic_dataset,
                                                        generator_name=bundle.configs['generator_name'],
                                                        synth_p=synth_p)
        trained_classifier = train_augmented_classifier(synth_training_set, val_set,
                                                        num_epochs, batch_size, learning_rate,
                                                        balance_classes,
                                                        model_type,
                                                        random_seed,
                                                        data_augment,
                                                        geometric_augment,
                                                        color_jitter,
                                                        h_flip_p,
                                                        v_flip_p,
                                                        synth_p=synth_p
                                                        )
        trained_classifiers.append(trained_classifier)
    log.info(f'Training baseline without synthetic data.')
    baseline = train_classifier(train_data=train_set, val_data=val_set, model_type=model_type, num_epochs=num_epochs, batch_size=batch_size,
                                learning_rate=learning_rate, balance_classes=balance_classes, random_seed=random_seed, data_augment=data_augment, geometric_augment=geometric_augment,
                                color_jitter=color_jitter, h_flip_p=h_flip_p, v_flip_p=v_flip_p, synth_p=synth_p)
    return TrainedClassifiersOnSynthData(baseline=baseline, classifiers=trained_classifiers)


@step
def evaluate_synth_trained_classifiers(trained_classifiers: TrainedClassifiersOnSynthData):
    raise NotImplementedError()
    # for classifier in trained_classifiers.classifiers:
        # TODO need the optimal threshold :) could put it in TrainedClassifier straight up
        # evaluate_classifier()