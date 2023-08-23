import os
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.train_test_split import TrainSet, create_train_test_val, TrainTestValBundle
from sample_augment.models.generator import GC10_CLASSES, StyleGANGenerator
from sample_augment.models.stylegan2.projector import load_model, run_projection
from sample_augment.data.synth_data import SynthData
from sample_augment.utils import log
from sample_augment.utils.path_utils import root_dir, shared_dir


def project_training_subset(data: TrainSet, generator_name: str, number_per_class=20):
    """
        Take the maximum number of images we can take from each class. So argmax.
        It will be around 40.
        We want to reach 50 per class so there will be some interpolation left to be done
    @param number_per_class:
    @param data:
    @param generator_name:
    @return:
    """
    G, device = load_model(network_pkl=str(root_dir / f'TrainedStyleGAN/{generator_name}.pkl'))
    # synthetic_imgs = []
    # synthetic_labels = []
    projected_indices = set()
    for class_idx in range(G.c_dim):  # c_dim == num_classes
        # we could try to be smarter here and hard-code some combinations we don't want
        # such as welding_line should not contain crescent gap, etc. since it muddies the results
        # sample from the training_set from each class
        class_indices = np.where(data.primary_label_tensor == class_idx)[0]
        sample_indices = np.random.choice(class_indices, size=number_per_class,
                                          replace=len(class_indices) < number_per_class)
        if len(class_indices) < number_per_class:
            log.warning(
                f"Not enough instances for class {GC10_CLASSES[class_idx]}, found {len(class_indices)}"
                f" but needed {number_per_class}")
        target_classname = GC10_CLASSES[class_idx]
        outdir = str(shared_dir / "projected" / generator_name)
        existing_files_count = len(list(Path(outdir).glob(f'{target_classname}_*_proj.npz')))
        log.info(f"Found {existing_files_count} existing projections for class {target_classname}")
        # Calculate the remaining number of projections required for this class
        remaining_number = max(number_per_class - existing_files_count, 0)
        if remaining_number == 0:
            log.info(f"All {number_per_class} projections for class {target_classname} already exist. Skipping.")
            continue

        log.info(f"Projecting class {target_classname}.")
        for idx in sample_indices:
            # Fetch image data and run projection
            target_image_data = data.image_tensor[idx].numpy()
            target_class_label = data.label_tensor[idx]

            name = f"{target_classname}_{idx}"
            file_path = Path(f'{outdir}/{name}_proj.npz')
            if file_path.exists():
                log.debug(f"Skipping {file_path}")
                continue
            else:
                log.debug(f"Generating for {file_path}")
            _synth_image = run_projection(G, device, target_image_data, target_class_label,
                                          name=f"{target_classname}_{idx}",
                                          outdir=str(shared_dir / "projected" / generator_name),
                                          seed=class_idx * number_per_class + idx,
                                          num_steps=1000)

            # synthetic_imgs.append(synth_image)
            # synthetic_labels.append(target_class_label)

        sample_indices_tuple = tuple(sample_indices)
        if sample_indices_tuple not in projected_indices:
            projected_indices.add(sample_indices_tuple)
        else:
            log.warning('duplicate indices')
            projected_indices.add(sample_indices_tuple)

    # synthetic_imgs_tensor = torch.stack(synthetic_imgs)
    # synthetic_labels_tensor = torch.stack(synthetic_labels)
    # return SynthData(synthetic_images=synthetic_imgs_tensor, synthetic_labels=synthetic_labels_tensor,
    #                  multi_label=True)


def main():
    random_seed = 100
    val_ratio = 0.1
    test_ratio = 0.2
    split_min_instances = 10
    generator_name = "wdataaug-028_012200"
    # if len(sys.argv) >= 2:
    #     log.info(f"Using sys.argv {generator_name}")
    #     generator_name = sys.argv[1]
    # else:
    log.info(f"Using default generator_name={generator_name}")
    # on colab this works apparently
    data: AugmentDataset = AugmentDataset.from_file(root_dir / "AugmentDataset/dataset_f00581.json")
    # could consider upgrading python to 3.8 for ease of use on DS machine
    # root = Path(r'E:\Master_Thesis_Nils\data\AugmentDataset')
    # training_tuple = (torch.load(str(root / 'f00581_primary_label_tensor.pt')),
    #                                                     torch.load(str(root / 'f00581_tensors_0.pt')),
    #                                                     torch.load(str(root / 'f00581_tensors_1.pt'))
    #                                                     )
    train_test_val: TrainTestValBundle = create_train_test_val(data, random_seed=random_seed, val_ratio=val_ratio,
                                                               test_ratio=test_ratio, min_instances=split_min_instances)
    project_training_subset(train_test_val.train, generator_name)


def from_projected_images(_training_set: TrainSet, generator_name: str, number_per_class=50):
    synthetic_imgs = []
    synthetic_labels = []

    generator = StyleGANGenerator.load_from_name(generator_name)
    class_to_latents_and_labels = defaultdict(list)
    outdir: Path = shared_dir / "projected" / generator_name
    assert outdir.is_dir(), f"can't find projected dir for generator {generator_name}"
    # Collect latent vectors and multilabel vectors for each primary class
    for file_name in os.listdir(outdir):
        if "_proj.npz" in file_name:
            primary_class_name = '_'.join(file_name.split('_')[:-2])
            npz_path = outdir / file_name
            with np.load(str(npz_path)) as data:
                target_latent = torch.tensor(data['w'])
                target_labels = torch.tensor(data['c'])
                class_to_latents_and_labels[primary_class_name].append((target_latent, target_labels))

    # Interpolate between random pairs of instances for each primary class
    for primary_class_name, latents_and_labels in class_to_latents_and_labels.items():
        original_count = len(latents_and_labels)
        assert original_count >= 2, f"Cannot interpolate with less than 2 instances for class {primary_class_name}"

        # Select random pairs without duplicates
        log.info(f"{primary_class_name}: interpolating {number_per_class - original_count} instances..")
        pairs_to_interpolate = random.sample(
            [(i, j) for i in range(original_count) for j in range(i + 1, original_count)],
            number_per_class - original_count)

        # Add original instances
        for latent, label in latents_and_labels:
            synth_img = generator.w_to_img(latent).squeeze().cpu().permute(2, 0, 1)
            # noinspection PyTypeChecker
            # synth_img = torch.tensor(np.array(synth_img)).permute(2, 0, 1)  # Change HWC to CHW
            synthetic_imgs.append(synth_img)
            synthetic_labels.append(label)

        # Interpolate new instances
        for idx1, idx2 in pairs_to_interpolate:
            alpha = random.uniform(0.2, 0.8)
            interpolated_latent = (1 - alpha) * latents_and_labels[idx1][0] + alpha * latents_and_labels[idx2][0]
            interpolated_label = (1 - alpha) * latents_and_labels[idx1][1] + alpha * latents_and_labels[idx2][1]
            synth_img = generator.w_to_img(interpolated_latent).squeeze().cpu().permute(2, 0, 1)  # Change HWC to CHW
            synthetic_imgs.append(synth_img)
            synthetic_labels.append(interpolated_label)

    synthetic_imgs_tensor = torch.stack(synthetic_imgs)
    synthetic_labels_tensor = torch.stack(synthetic_labels)
    return SynthData(synthetic_images=synthetic_imgs_tensor, synthetic_labels=synthetic_labels_tensor, multi_label=True)


if __name__ == '__main__':
    main()
