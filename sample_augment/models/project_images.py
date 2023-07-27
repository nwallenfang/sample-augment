import sys

import numpy as np

from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.train_test_split import TrainSet, create_train_test_val, TrainTestValBundle
from sample_augment.models.stylegan2.projector import run_projection, load_model
from sample_augment.utils.path_utils import root_dir, shared_dir

from sample_augment.models.generator import GC10_CLASSES


def project_training_subset(training_set: TrainSet, generator_name: str, number_per_class=20):
    G, device = load_model(network_pkl=str(root_dir / f'TrainedStyleGAN/{generator_name}.pkl'))

    for class_idx in range(10):
        # TODO we'll probably need to be smarter here and hard-code some combinations we don't want
        #   such as welding_line should not contain crescent gap, etc. since it muddies the results
        # sample from the training_set from each class
        class_indices = np.where(training_set.primary_label_tensor == class_idx)[0]
        sample_indices = np.random.choice(class_indices, size=number_per_class, replace=False)

        for instance_idx in sample_indices:
            # Fetch image data and run projection
            target_image_data = training_set.image_tensor[instance_idx].numpy()
            target_class_label = training_set.label_tensor[instance_idx]  # TODO put into projector
            target_classname = GC10_CLASSES[class_idx]
            projected_fname = f"{target_classname}_{instance_idx}"

            # Assuming run_projection is updated to accept a numpy array as target image data
            run_projection(
                G,
                device,
                target_image_data,
                target_class_label,
                target_classname,
                projected_fname,
                outdir=str(shared_dir / "projected"),
                save_video=False,
                seed=1,
                num_steps=1000
            )


def main():
    random_seed = 100
    val_ratio = 0.1
    test_ratio = 0.2
    split_min_instances = 10
    instances_per_class = 20
    assert len(sys.argv) >= 2, "generator name needed as sys arg"
    generator_name = sys.argv[1]
    data: AugmentDataset = AugmentDataset.from_file(root_dir / "AugmentDataset/dataset_f00581.json")
    train_test_val: TrainTestValBundle = create_train_test_val(data, random_seed=random_seed, val_ratio=val_ratio,
                                                               test_ratio=test_ratio, min_instances=split_min_instances)
    project_training_subset(train_test_val.train, generator_name)


if __name__ == '__main__':
    main()
