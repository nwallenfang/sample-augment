

import numpy as np

from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.train_test_split import TrainSet, create_train_test_val, TrainTestValBundle
from sample_augment.models.stylegan2.projector import run_projection, load_model
from sample_augment.utils.path_utils import root_dir, shared_dir

from sample_augment.models.generator import GC10_CLASSES


def project_training_subset(training_set: TrainSet, number_per_class=20):
    G, device = load_model(network_pkl=str(root_dir / 'TrainedStyleGAN/apa-020_004400.pkl'))

    for class_idx in range(10):
        # sample from the training_set from each class
        # TODO we'll probably need to be smarter here and hard-code some combinations we don't want
        #   such as welding_line should not contain crescent gap, etc. since it muddies the results
        class_indices = np.where(training_set.primary_label_tensor == class_idx)[0]
        sample_indices = np.random.choice(class_indices, size=number_per_class, replace=False)

        for instance_idx in sample_indices:
            # Fetch image data and run projection
            target_image_data = training_set.image_tensor[instance_idx].numpy()
            target_classname = GC10_CLASSES[class_idx]
            projected_fname = f"proj_{target_classname}_{instance_idx}"

            # Assuming run_projection is updated to accept a numpy array as target image data
            run_projection(
                G,
                device,
                target_image_data,
                class_idx,
                target_classname,
                projected_fname,
                outdir=str(shared_dir / "projected"),
                save_video=False,
                seed=1,
                num_steps=1000
            )


if __name__ == '__main__':
    random_seed = 100
    val_ratio = 0.1
    test_ratio = 0.2
    min_instances = 10
    data: AugmentDataset = AugmentDataset.from_file(root_dir / "AugmentDataset/dataset_f00581.json")
    train_test_val: TrainTestValBundle = create_train_test_val(data, random_seed=random_seed, val_ratio=val_ratio,
                                                               test_ratio=test_ratio, min_instances=min_instances)
    project_training_subset(train_test_val.train)

    # pkl_path = root_dir / 'TrainedStyleGAN/network-snapshot-001000.pkl'
    # generator = StyleGANGenerator(pkl_path=pkl_path)

    # if len(sys.argv) > 1:
    #     img_path = sys.argv[1]
    # else:
    #     img_path = shared_dir / "gc10" / "01" / "img_02_425506300_00018.jpg"
    #     log.info(f"using default img_path {img_path}")
    #
    # prefix = shared_dir / "gc10"
    # img_paths = {
    #     0: "01/img_03_4406742300_00001.jpg",  # punching_hole
    #     1: "02/img_02_3402616900_00001.jpg",  # welding line
    #     2: "03/img_01_425008500_00874.jpg",  # crescent gap
    #     7: "08/img_03_4402329000_00001.jpg"  # waist folding
    # }
    # for target_class, img_path in img_paths.items():
    #     log.info(f'--- {GC10_CLASSES[target_class]} ---')
    #     project(pkl_path=root_dir / 'TrainedStyleGAN/apa-020_004400.pkl', image_path=prefix / img_path,
    #             target_class=target_class)
