from matplotlib import pyplot as plt
from torchvision.transforms import ToPILImage

from sample_augment.core import step
from sample_augment.core.config import read_config
from sample_augment.core.experiment import Experiment
from sample_augment.core.step import find_steps
from sample_augment.data.synth_data import SyntheticBundle
from sample_augment.utils.path_utils import root_dir


def check_for_duplicates(images):
    seen = set()
    duplicates = []

    for i, img in enumerate(images):
        img_flat_str = str(img.cpu().numpy().flatten())
        if img_flat_str in seen:
            print(f"Duplicate found at index {i}")
            duplicates.append(i)
        else:
            seen.add(img_flat_str)

    if not duplicates:
        print("No duplicates found.")
    else:
        print(f"Found duplicates at these indices: {duplicates}")


@step
def look_at_synth_bundle(bundle: SyntheticBundle):
    limit = 20
    to_pil = ToPILImage()
    for strat, synthetic_data in zip(bundle.configs['strategies'], bundle.synthetic_datasets):
        print(f"-- {strat} --")
        imgs = synthetic_data.synthetic_images
        check_for_duplicates(imgs)
        for i in range(min(limit, len(imgs))):
            img = imgs[i]
            print(img.shape)

            img_pil = to_pil(img.cpu())

            plt.figure()
            plt.imshow(img_pil)
            plt.title(f"{strat} - Image {i + 1}")
            plt.show()


def main():
    """
        CLI for running experiments concerning
    """
    config = read_config(root_dir.parent / "config.json")
    find_steps(include=['test', 'data', 'models', 'sampling'], exclude=['models.stylegan2'])
    # create Experiment instance
    experiment = Experiment(config)

    # evaluate_k_classifiers, k_fold_plot_loss_over_epochs, imagefolder_to_tensors, k_fold_train_classifier
    # experiment.run("train_augmented_classifier")
    # experiment.run("evaluate_classifier")
    # experiment.run('look_at_augmented_train_set')
    experiment.run("create_synthetic_bundle")


if __name__ == '__main__':
    main()
