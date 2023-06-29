import json
from pathlib import Path
from typing import Dict

import numpy as np

from sample_augment.core import step
from sample_augment.models.train_classifier import ClassifierMetrics
from sample_augment.utils import log
from sample_augment.utils.plot import prepare_latex_plot


def read_f1_losses(kfold_json: Dict) -> np.ndarray:
    f1_scores = []
    # num_epochs = clas
    for classifier_json in kfold_json['classifiers']:
        metrics = ClassifierMetrics.from_dict(classifier_json['metrics'])
        f1_scores.append(metrics.validation_f1)
    return np.stack(f1_scores)


# TODO turn into a proper step
@step
def compare_f1_scores(shared_directory: Path):
    import matplotlib.pyplot as plt
    import matplotlib

    # Load JSONs
    kfold_augmented_json = json.load(
        open('/home/nils/thesis/sample-augment/data/KFoldTrainedClassifiers/aug-01_ecc814.json'))
    kfold_no_aug_json = json.load(
        open('/home/nils/thesis/sample-augment/data/KFoldTrainedClassifiers/baseline-noaug_58f246.json'))

    # Extract F1 scores
    f1_aug = read_f1_losses(kfold_augmented_json)
    f1_noaug = read_f1_losses(kfold_no_aug_json)

    # Compute mean and standard deviation
    f1_aug_mean = np.mean(f1_aug, axis=0)
    f1_noaug_mean = np.mean(f1_noaug, axis=0)
    f1_aug_std = np.std(f1_aug, axis=0)
    f1_noaug_std = np.std(f1_noaug, axis=0)

    # Number of epochs
    epochs = range(1, len(f1_aug_mean) + 1)

    # Set the font to be serif, rather than sans
    matplotlib.rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
    # Use LaTeX to handle all text layout
    matplotlib.rc('text', usetex=True)
    # Ensure that matplotlib's LaTeX output matches the LaTeX document
    matplotlib.rc('figure', dpi=200)
    matplotlib.rcParams.update({'font.size': 14})

    # Create the plot
    plt.figure(figsize=(10, 7))
    x_ticks = np.arange(1, max(epochs) + 1, 5)
    x_ticks = np.insert(x_ticks, 1, 1)  # Insert 1 at the beginning

    prepare_latex_plot()

    # F1 for each classifier in light color
    for i in range(f1_aug.shape[0]):
        plt.plot(epochs, f1_aug[i], color='blue', alpha=0.1)
        plt.plot(epochs, f1_noaug[i], color='red', alpha=0.1)

    # Average F1
    plt.plot(epochs, f1_aug_mean, label='Mean F1 (Augmented)', color='blue', linewidth=2)
    plt.fill_between(epochs, f1_aug_mean - f1_aug_std, f1_aug_mean + f1_aug_std, color='blue', alpha=0.2)

    plt.plot(epochs, f1_noaug_mean, label='Mean F1 (Non-Augmented)', color='red', linewidth=2)
    plt.fill_between(epochs, f1_noaug_mean - f1_noaug_std, f1_noaug_mean + f1_noaug_std, color='red',
                     alpha=0.2)

    # Labels, title and legend
    plt.xlabel('Epoch')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.xlim([1, len(f1_aug_mean)])
    plt.ylim([0.0, 1.0])  # F1 score range from 0 to 1
    plt.xticks(x_ticks)
    plt.yticks(np.arange(0, 1.1, 0.1))  # Show y-ticks from 0 to 1 with a step of 0.1

    figure_path = shared_directory / "f1_comparison.pdf"
    log.info(f"Save F1 Comparison plot to {figure_path}")
    plt.savefig(figure_path, bbox_inches='tight', format='pdf')


if __name__ == '__main__':
    compare_f1_scores()
    # kfold_trained_path = Path(r"C:\Users\Nils\Documents\Masterarbeit\sample-augment\data"
    #                           r"\KFoldTrainedClassifiers\aug-01_ecc814.json")
    # with open(kfold_trained_path) as kfold_trained_file:
    #     kfold_trained_data = json.load(kfold_trained_file)
    #
    # k_fold_plot_loss_over_epochs(KFoldTrainedClassifiers.from_dict(kfold_trained_data),
    #                              Path(r"C:\Users\Nils\Documents\Masterarbeit\sample-augment\data\shared"),
    #                              "aug-01")
