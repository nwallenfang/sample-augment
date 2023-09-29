from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from torch import Tensor

from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.train_test_split import create_train_test_val
from sample_augment.models.evaluate_classifier import ValidationPredictions, predict_validation_set
from sample_augment.models.generator import GC10_CLASSES
from sample_augment.models.train_classifier import TrainedClassifier, determine_threshold_vector
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir
from sample_augment.utils.plot import prepare_latex_plot


def class_get_positive_and_negative_instances(labels: Tensor, predictions: ValidationPredictions, class_idx: int):
    """
    Extract and segregate the predicted probabilities into positive and negative instances based on true labels.
    """
    probs = (predictions.predictions[:, class_idx]).cpu().numpy()
    positive_probs = probs[labels[:, class_idx] == 1]
    negative_probs = probs[labels[:, class_idx] == 0]

    return positive_probs, negative_probs


def plot_histograms_by_class(df: pd.DataFrame):
    bin_edges = np.linspace(0, 1, 12)  # Creates 30 bins from 0 to 1

    for class_name in GC10_CLASSES:
        subset = df[df['Class'] == class_name]

        positives = subset[subset['Type'] == 'Positive']['Probability']
        negatives = subset[subset['Type'] == 'Negative']['Probability']

        plt.figure(figsize=(10, 5))
        plt.hist(positives, bins=bin_edges, alpha=0.5, label='Positive', color='blue', density=True)
        plt.hist(negatives, bins=bin_edges, alpha=0.5, label='Negative', color='red', density=True)

        plt.title(f"Histogram for Class: {class_name}")
        plt.xlabel("Classification Probability")
        plt.ylabel("Density")
        plt.legend(loc='upper right')

        plt.show()


def plot_first_n_histograms(df: pd.DataFrame, lambda_thresholds: Tuple[Tensor, Tensor]):
    deep_colors = sns.color_palette("deep")

    plt.figure(figsize=(15, 5))
    idx = [7, 8, 9]
    n = len(idx)
    for i, class_idx in enumerate(idx):
        class_name = GC10_CLASSES[class_idx]
        plt.subplot(1, n, i + 1)

        subset = df[df['Class'] == class_name]

        positives = subset[subset['Type'] == 'Positive']['Probability']
        negatives = subset[subset['Type'] == 'Negative']['Probability']

        bin_edges = np.linspace(0, 1, 12)

        plt.hist([positives, negatives], bins=bin_edges, color=[deep_colors[0], deep_colors[1]], label=['Positive',
                                                                                                        'Negative'],
                 align='left', density=True)
        # plt.hist(positives, bins=bin_edges, alpha=0.7, label='Positive', color=deep_colors[0], density=True)
        # plt.hist(negatives, bins=bin_edges, alpha=0.7, label='Negative', color=deep_colors[1], density=True)

        lambda_0_thres = lambda_thresholds[0][class_idx].cpu().numpy()
        lambda_04_thres = lambda_thresholds[1][class_idx].cpu().numpy()

        plt.axvline(x=lambda_0_thres, color=deep_colors[7], linestyle='--', label=f'λ=0: {lambda_0_thres:.2f}',
                    linewidth=2.0)
        plt.axvline(x=lambda_04_thres, color=deep_colors[3], linestyle='--', label=f'λ=0.4: {lambda_04_thres:.2f}',
                    linewidth=2.0)

        plt.title(f"Histogram for Class: {class_name}")
        plt.xlabel("Classification Probability")
        plt.ylabel("Density")
        plt.legend()

    plt.tight_layout()
    plt.show()


def plot_two_sided_violin(from_csv: Optional[Path] = None, plot=True):
    """
    Create a two-sided violin plot comparing the distribution of classification outputs for
    positive and negative instances.
    """
    # TODO maybe keep it simple: positive boxplot vs negative boxplot
    if not from_csv:
        classifier = TrainedClassifier.from_name("ViT-100e_ce6b40")
        dataset = AugmentDataset.from_name("dataset_f00581")
        val_set = create_train_test_val(dataset, random_seed=100, test_ratio=0.1, val_ratio=0.2, min_instances=10).val
        predictions: ValidationPredictions = predict_validation_set(classifier, val_set, batch_size=32)
        # Convert list of probabilities to dataframe
        data = []
        for i, class_name in enumerate(GC10_CLASSES):
            print(f"-- {class_name} --")
            pos_probs, neg_probs = class_get_positive_and_negative_instances(val_set.label_tensor, predictions, i)

            for prob in pos_probs:
                data.append({'Class': class_name, 'Probability': prob, 'Type': 'Positive'})

            for prob in neg_probs:
                data.append({'Class': class_name, 'Probability': prob, 'Type': 'Negative'})

        lambda_0_thresholds = determine_threshold_vector(predictions.predictions, val_set, 0.0, n_support=250)
        lambda_04_thresholds = determine_threshold_vector(predictions.predictions, val_set, 0.4, n_support=250)

        # Save the thresholds as .npy files
        np.save(shared_dir / "lambda_0_thresholds.npy", lambda_0_thresholds.cpu().numpy())
        np.save(shared_dir / "lambda_04_thresholds.npy", lambda_04_thresholds.cpu().numpy())

        df = pd.DataFrame(data)
    else:
        log.info('from csv')
        # from csv
        df = pd.read_csv(from_csv)
        lambda_0_thresholds = torch.tensor(np.load(shared_dir / "lambda_0_thresholds.npy"))
        lambda_04_thresholds = torch.tensor(np.load(shared_dir / "lambda_04_thresholds.npy"))
        # plot_histograms_by_class(df)
        plot_first_n_histograms(df, lambda_thresholds=(lambda_0_thresholds, lambda_04_thresholds))
        return

    if plot:
        # Create the violin plot
        prepare_latex_plot()
        plt.figure(figsize=(12, 6))
        # sns.violinplot(x='Class', y='Probability', hue='Type', data=df, split=True, cut=0, inner=None, alpha=0.7,
        #                palette="deep", scale='width')
        sns.stripplot(x='Class', y='Probability', hue='Type', data=df, palette="deep", size=2, jitter=True)
        plt.xticks(rotation=90)
        plt.ylabel(f"Classification Probability")  # Assuming classifier.name gives the model name

        plt.savefig(shared_dir / "figures" / f"thresholding_visualization.pdf",
                    bbox_inches="tight")

    df.to_csv(shared_dir / f"thresholding_visualization.csv")


if __name__ == '__main__':
    plot_two_sided_violin(shared_dir / f"thresholding_visualization.csv")
    # plot_two_sided_violin()
