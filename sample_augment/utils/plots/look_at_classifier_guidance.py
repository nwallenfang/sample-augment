import os
from typing import Tuple, List

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt

from sample_augment.sampling.classifier_guidance import GuidanceMetric
from sample_augment.utils.path_utils import shared_dir
from sample_augment.utils.plot import prepare_latex_plot, multihot_to_classnames


def load_guidance_metric_data(guidance_metric: str,
                              output_dir=shared_dir / 'generated/cguidance') -> Tuple[
    List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    n_combinations = 3  # As currently set in your code
    loaded_metrics = []
    loaded_top_images = []
    loaded_bottom_images = []
    loaded_labels = []

    for idx in range(n_combinations):
        # Load Metrics
        metric_path = os.path.join(output_dir, f'metric_{guidance_metric}_comb_{idx}.npy')
        loaded_metric = np.load(metric_path)
        loaded_metrics.append(loaded_metric)

        # Load Top Images
        top_images = []
        for i in range(1, 7):  # We saved the top 6 images
            img_path = os.path.join(output_dir, f'top_{i}_{guidance_metric}_comb_{idx}.png')
            img = np.array(Image.open(img_path)) / 255.0
            top_images.append(img)
        loaded_top_images.append(np.array(top_images))

        # Load Bottom Images
        bottom_images = []
        for i in range(1, 7):  # We saved the bottom 6 images
            img_path = os.path.join(output_dir, f'bottom_{i}_{guidance_metric}_comb_{idx}.png')
            img = np.array(Image.open(img_path)) / 255.0
            bottom_images.append(img)
        loaded_bottom_images.append(np.array(bottom_images))

        # Load Labels
        label_path = os.path.join(output_dir, f'{guidance_metric}_comb_{idx}.npy')
        loaded_label = np.load(label_path)
        loaded_labels.append(loaded_label)

    return loaded_metrics, loaded_top_images, loaded_bottom_images, loaded_labels


def plot_image_grid(name: str, metrics: List[np.ndarray], top_imgs: List[np.ndarray], bot_imgs: List[np.ndarray],
                    labels):
    # n_combinations = len(metrics)
    short = "L_2" if name == 'L2 Distance' else "H"
    prepare_latex_plot()

    for idx, label in enumerate(labels):
        # Extract data for this combination
        metric = metrics[idx]
        top_images = top_imgs[idx]
        bot_images = bot_imgs[idx]

        # Sort by metric
        top_idx = np.argsort(metric)[-4:]  # Assuming 6 top images
        bot_idx = np.argsort(metric)[:4]  # Assuming 6 bottom images

        fig, axs = plt.subplots(2, 4, figsize=(0.7 * 10, 0.7 * 6.5))

        for i, image_idx in enumerate(bot_idx):
            axs[0, i].imshow(bot_images[i])
            axs[0, i].set_title(f"${short} = {metric[image_idx]:.2f}$")
            axs[0, i].axis('off')
            # if i == 0:
            #     axs[0, i].text(-128, 100, "Lowest", fontsize=12, verticalalignment='center',
            #                    horizontalalignment='center')
        for i, image_idx in enumerate(reversed(top_idx)):
            axs[1, i].imshow(top_images[i])
            axs[1, i].set_title(f"${short} = {metric[image_idx]:.2f}$")
            axs[1, i].axis('off')
            # if i == 0:
            #     axs[1, i].text(-128, 100, "Highest", fontsize=12, verticalalignment='center',
            #                    horizontalalignment='center')

        # classes = multihot_to_classnames(label)
        # plt.suptitle(r"Classes: \texttt{" + classes[0] + r"} and \texttt{" + classes[1] + "}")
        plt.tight_layout()
        plt.savefig(shared_dir / 'generated' / 'cguidance' / f'grid_{name}_{idx}.pdf', bbox_inches='tight')


def plot_metric_histogram(name: str, metrics: np.ndarray):
    plt.figure(figsize=(10, 6))

    # Concatenate all the metrics from different combinations to form a single array
    all_metrics = np.concatenate(metrics)

    plt.hist(all_metrics, bins=18, alpha=0.7, color='blue', edgecolor='black')

    # plt.title(title)
    plt.xlabel(f'{name}')
    plt.ylabel('Frequency')

    plt.show()


if __name__ == '__main__':
    # noinspection PyTupleAssignmentBalance
    metrics, top_imgs, bot_imgs, labels = load_guidance_metric_data(GuidanceMetric.L2Distance.__name__)
    # plot_metric_histogram("L2 Distance", metrics)
    plot_image_grid("L2 Distance", metrics, top_imgs, bot_imgs, labels)
    metrics, top_imgs, bot_imgs, labels = load_guidance_metric_data(GuidanceMetric.Entropy.__name__)
    plot_image_grid("Entropy", metrics, top_imgs, bot_imgs, labels)

    # plot_metric_histogram("Entropy", metrics)
