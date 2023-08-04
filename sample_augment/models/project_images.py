import sys

import numpy as np

from sample_augment.models.stylegan2.projector_stylegan3_fun import run_projection_sgan3_fun
from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.train_test_split import TrainSet, create_train_test_val, TrainTestValBundle
from sample_augment.models.generator import GC10_CLASSES
from sample_augment.models.stylegan2.projector import load_model
from sample_augment.utils.path_utils import root_dir, shared_dir


def project_training_subset(data: TrainSet, generator_name: str, number_per_class=20):
    G, device = load_model(network_pkl=str(root_dir / f'TrainedStyleGAN/{generator_name}.pkl'))
    projected_indices = set()
    for class_idx in range(G.c_dim):  # c_dim == num_classes
        # we could try to be smarter here and hard-code some combinations we don't want
        # such as welding_line should not contain crescent gap, etc. since it muddies the results
        # sample from the training_set from each class
        class_indices = np.where(data.primary_label_tensor == class_idx)[0]
        sample_indices = np.random.choice(class_indices, size=number_per_class,
                                          replace=len(class_indices) < number_per_class)

        # TODO how can it be that the same picture (index) got used for two different primary classes??
        #  for context it's 488 for welding_line and crescent_gap
        for idx in sample_indices:
            # Fetch image data and run projection
            target_image_data = data.image_tensor[idx].numpy()
            target_class_label = data.label_tensor[idx]
            target_classname = GC10_CLASSES[class_idx]

            run_projection_sgan3_fun(
                G,
                target_image_data,
                target_class_label,
                identifier=f"{target_classname}_{idx}",
                seed=class_idx * number_per_class + idx,
                # ensure that different seeds get used for all instances of the same class
                # (might have same target image multiple times for small classes)
                out_dir=shared_dir / "projected",
                num_steps=1000
            )
        projected_indices.add(sample_indices)
        # run_projection(
        #     G,
        #     device,
        #     target_image_data,
        #     target_class_label,
        #     target_classname,
        #     projected_fname,
        #     outdir=str(shared_dir / "projected"),
        #     save_video=False,
        #     # ensure that different seeds get used for all instances of the same class
        #     # (might have same target image multiple times for small classes)
        #     seed=instance_idx,
        #     num_steps=1000
        # )


def main():
    random_seed = 100
    val_ratio = 0.1
    test_ratio = 0.2
    split_min_instances = 10
    assert len(sys.argv) >= 2, "generator name needed as sys arg"
    generator_name = sys.argv[1]
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
