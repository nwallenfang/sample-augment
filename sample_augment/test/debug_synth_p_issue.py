import numpy as np
from matplotlib import pyplot as plt

from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.synth_data import SyntheticBundle, SynthAugmentedTrain, SynthData
from sample_augment.data.train_test_split import create_train_test_val, TrainSet, TrainTestValBundle
from sample_augment.models.train_classifier import train_classifier, train_augmented_classifier
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir


def visualize_samples(dataset, num_samples=15, times=3):
    num_rows = 3
    num_cols = num_samples // 3

    for time in range(times):
        time_offs = num_samples * time
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))
        for i in range(num_samples):
            row = i // num_cols
            col = i % num_cols
            # index = np.random.randint(0, len(dataset))
            image, label = dataset[time_offs + row * num_cols + col]
            # Assuming the image is a single-channel (grayscale) or 3-channel (RGB) image
            if image.shape[0] == 1:
                image = image.squeeze(0)
            else:
                image = image.permute(1, 2, 0)
            axes[row, col].imshow(image.cpu().numpy(), cmap='gray' if image.shape[0] == 1 else None)
            axes[row, col].set_title(f"Label: {label}")
            axes[row, col].axis('off')
        plt.suptitle(dataset.__full_name__)
        plt.show()
        plt.close(fig)


def main():
    # reading synthetic data and train/test/val
    synth_bundle = SyntheticBundle.from_name('synthbundle_c83f5b')
    strategies = synth_bundle.configs['strategies']
    synth_data: SynthData = synth_bundle.synthetic_datasets[0]
    dataset = AugmentDataset.from_name('dataset_f00581')
    bundle: TrainTestValBundle = create_train_test_val(dataset, random_seed=100, test_ratio=0.2, val_ratio=0.1,
                                                       min_instances=10)

    log.info(f'trying on dataset from strategy {strategies[0]}')

    # experiment constants, basically
    generator_name = 'synth-p-debug'
    generated_dir = shared_dir / "generated" / "synth-p-debug"
    synth_p = 0.0

    # defining training sets to compare
    baseline_dataset: TrainSet = bundle.train
    synth_dataset: SynthAugmentedTrain = SynthAugmentedTrain(name=f"synth-aug-{generator_name}", root_dir=generated_dir,
                                                             img_ids=baseline_dataset.img_ids,
                                                             tensors=(
                                                             baseline_dataset.tensors[0], baseline_dataset.tensors[1]),
                                                             primary_label_tensor=baseline_dataset.primary_label_tensor,
                                                             synthetic_images=synth_data.synthetic_images,
                                                             synthetic_labels=synth_data.synthetic_labels,
                                                             synth_p=synth_p,
                                                             multi_label=synth_data.multi_label)

    visualize_samples(baseline_dataset)
    visualize_samples(synth_dataset)

    # TODO need to fill these methods with the correct params
    # train_classifier()
    # train_augmented_classifier()


if __name__ == '__main__':
    main()
