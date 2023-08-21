import os
import sys

import numpy as np
import torch
from PIL import Image

from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.train_test_split import TrainSet, create_train_test_val, TrainTestValBundle
from sample_augment.models.generator import GC10_CLASSES
from sample_augment.models.stylegan2.projector import load_model, run_projection
from sample_augment.sampling.synth_augment import SynthData
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
    synthetic_imgs = []
    synthetic_labels = []
    projected_indices = set()
    for class_idx in range(G.c_dim):  # c_dim == num_classes
        # we could try to be smarter here and hard-code some combinations we don't want
        # such as welding_line should not contain crescent gap, etc. since it muddies the results
        # sample from the training_set from each class
        class_indices = np.where(data.primary_label_tensor == class_idx)[0]
        sample_indices = np.random.choice(class_indices, size=number_per_class,
                                          replace=len(class_indices) < number_per_class)

        # TODO save label in npz, and then we run this by itself and implement create_projected_synthdata()
        for idx in sample_indices:
            # Fetch image data and run projection
            target_image_data = data.image_tensor[idx].numpy()
            target_class_label = data.label_tensor[idx]
            target_classname = GC10_CLASSES[class_idx]

            synth_image = run_projection(G, device, target_image_data, target_class_label,
                                         name=f"{target_classname}_{idx}",
                                         outdir=str(shared_dir / "projected" / generator_name),
                                         seed=class_idx * number_per_class + idx,
                                         num_steps=1000)

            synthetic_imgs.append(synth_image)
            synthetic_labels.append(target_class_label)
            # run_projection_sgan3_fun(
            #     G,
            #     target_image_data,
            #     target_class_label,
            #     identifier=f"{target_classname}_{idx}",
            #     seed=class_idx * number_per_class + idx,
            #     # ensure that different seeds get used for all instances of the same class
            #     # (might have same target image multiple times for small classes)
            #     out_dir=shared_dir / "projected",
            #     num_steps=1000
            # )

        if sample_indices not in projected_indices:
            projected_indices.add(sample_indices)
        else:
            log.warning('duplicate indices')
            projected_indices.add(sample_indices)

    synthetic_imgs_tensor = torch.stack(synthetic_imgs)
    synthetic_labels_tensor = torch.stack(synthetic_labels)
    return SynthData(synthetic_images=synthetic_imgs_tensor, synthetic_labels=synthetic_labels_tensor,
                     multi_label=True)


def from_projected_images(_training_set: TrainSet, _generator_name: str, outdir: str, number_per_class=20):
    synthetic_imgs = []
    synthetic_labels = []

    # Assuming the file naming follows the pattern: *_proj.png for synthetic images and *_proj.npz for labels
    for class_idx in range(number_per_class):
        for file_name in os.listdir(outdir):
            if file_name.endswith(f"{class_idx}_proj.png"):
                # Read synthetic image
                synth_img_path = os.path.join(outdir, file_name)
                synth_img = Image.open(synth_img_path)
                # noinspection PyTypeChecker
                synth_img = torch.tensor(np.array(synth_img)).permute(2, 0, 1)  # Change HWC to CHW

                # Read corresponding label
                label_path = os.path.join(outdir, file_name.replace('_proj.png', '_proj.npz'))
                with np.load(label_path) as data:
                    target_class = torch.tensor(data['c'])

                synthetic_imgs.append(synth_img)
                synthetic_labels.append(target_class)

    synthetic_imgs_tensor = torch.stack(synthetic_imgs)
    synthetic_labels_tensor = torch.stack(synthetic_labels)
    return SynthData(synthetic_images=synthetic_imgs_tensor, synthetic_labels=synthetic_labels_tensor, multi_label=True)


def main():
    random_seed = 100
    val_ratio = 0.1
    test_ratio = 0.2
    split_min_instances = 10
    generator_name = "wdataaug-028_012200"
    if len(sys.argv) >= 2:
        generator_name = sys.argv[1]
    else:
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


if __name__ == '__main__':
    main()
