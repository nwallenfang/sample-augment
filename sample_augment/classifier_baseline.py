import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import matplotlib
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score

from sample_augment.core import step, Experiment, Store
from sample_augment.core.config import read_config
from sample_augment.core.step import find_steps
from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.train_test_split import create_train_test_val
from sample_augment.models.evaluate_classifier import KFoldClassificationReport
from sample_augment.models.evaluate_classifier import predict_validation_set
from sample_augment.models.train_classifier import ClassifierMetrics, TrainedClassifier, KFoldTrainedClassifiers
from sample_augment.utils import log
from sample_augment.utils.path_utils import root_dir
from sample_augment.utils.plot import prepare_latex_plot


def read_f1_losses(kfold_json: Dict) -> np.ndarray:
    f1_scores = []
    # num_epochs = clas
    for classifier_json in kfold_json['classifiers']:
        metrics = ClassifierMetrics.from_dict(classifier_json['metrics'])
        f1_scores.append(metrics.validation_f1)
    return np.stack(f1_scores)


@step
def compare_f1_scores(shared_directory: Path):

    # Load JSONs
    kfold_augmented_json = json.load(
        open('/home/nils/thesis/sample-augment/data/KFoldTrainedClassifiers/aug-01_ecc814.json'))
    kfold_no_aug_json = json.load(
        open('/home/nils/thesis/sample-augment/data/KFoldTrainedClassifiers/baseline-configs-noaug_58f246.json'))

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


def running_on_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False


experiment_to_step = {
    'baseline': 'evaluate_k_classifiers',
    'architecture': 'evaluate_k_classifiers',
    'sampling': 'evaluate_synth_trained_classifiers'
}


def run_classifier_experiment(experiment_name, limit=None):
    experiments_dir = root_dir.parent / 'experiments'
    if not running_on_colab():  # find_steps is called in init on colab
        find_steps(include=['test', 'data', 'models', 'sampling'], exclude=['models.stylegan2'])

    # Determine the paths based on the experiment name
    config_path_name = f"{experiment_name}-configs"
    run_path_name = f"{experiment_name}-runs"
    classifier_configs = experiments_dir / config_path_name

    count = 0
    for config_filename in sorted(os.listdir(classifier_configs)):
        if limit and count >= limit:
            log.info(f"limit of {limit} runs reached.")
            break

        config_path = classifier_configs / config_filename
        config = read_config(config_path)
        store_save_path = experiments_dir / run_path_name / f"{config.filename}.json"

        if store_save_path.exists():
            log.info(f"- skipping experiment {config_filename} -")
            continue
        else:
            log.info(f"- running experiment {config_filename} -")
        # create Experiment instance
        experiment = Experiment(config)
        experiment.run(experiment_to_step[experiment_name], store_save_path=store_save_path)
        del experiment

        count += 1


def evaluate_classifier_experiment(experiment_name: str):
    experiment_run_dir = root_dir.parent / "experiments" / f"{experiment_name}-runs"
    assert experiment_run_dir.exists(), f"{experiment_run_dir} does not exist."

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # collect experiment artifacts, care this takes tons of RAM like this preloading everything
    reports: Dict[str, KFoldClassificationReport] = {}  # run name to run json
    for run_filename in os.listdir(experiment_run_dir):
        name = run_filename.split("_")[0]
        artifacts = Store.load_from(experiment_run_dir / run_filename).artifacts
        reports[name] = artifacts['KFoldClassificationReport']

    all_macro_f1 = []
    all_auc = []

    baseline_config = read_config(root_dir.parent / "experiments" / f"{experiment_name}-configs" / "b00-baseline.json")

    augment_dataset: AugmentDataset = AugmentDataset.from_name('dataset_f00581')
    bundle = create_train_test_val(augment_dataset, baseline_config['random_seed'], baseline_config['test_ratio'],
                                   baseline_config['val_ratio'], baseline_config['min_instances'])
    val_set = bundle.val
    # TODO include this plot: class_performance_boxplot()
    # TODO make the evaluation procedure so TrainedClassifiers only get loaded if absolutely necessary
    # TODO make strip plot over the folds and their macro F1 as well maybe
    # TODO make loss plot for high learning rate vs low learning rate

    for run_name, run_data in reports.items():
        # noinspection PyUnresolvedReferences
        trained_class: List[TrainedClassifier] = run_data[0].classifiers
        val_losses = [clf.metrics.validation_loss for clf in trained_class]
        plt.plot(val_losses, label=f"{run_name} Validation Loss")

        macro_f1_scores = []
        auc_scores = []
        if not torch.cuda.is_available():
            log.info('no cuda - no auc metric')
        for classifier, class_report in zip(run_data[0].classifiers, run_data[1].reports):
            report = class_report.report
            macro_f1_scores.append(report['macro avg']['f1-score'])

            predictions = predict_validation_set(classifier, val_set, batch_size=32).predictions
            auc = roc_auc_score(val_set.label_tensor.numpy(), predictions, average='macro', multi_class='ovr')
            auc_scores.append(auc)

        avg_macro_f1 = np.mean(macro_f1_scores)
        avg_auc = np.mean(auc_scores)

        log.info(f"{run_name}: Macro avg F1: {avg_macro_f1}, Avg AUC: {avg_auc}")

        all_macro_f1.append(avg_macro_f1)
        all_auc.append(avg_auc)

    plt.legend()
    plt.title("Validation Losses Across Runs")
    plt.show()


def main():
    if len(sys.argv) > 1:
        assert len(sys.argv) == 3, 'run/eval {name}'
        if sys.argv[1] == 'run':
            run_classifier_experiment(sys.argv[2])
        elif sys.argv[1] == 'eval':
            evaluate_classifier_experiment(sys.argv[2])
        else:
            assert False, 'run/eval {name}'
    run_classifier_experiment("baseline", limit=1)
    # evaluate_classifier_experiment("baseline")
    # run_classifier_experiment("architecture")


if __name__ == '__main__':
    main()
    # compare_f1_scores()
    # kfold_trained_path = Path(r"C:\Users\Nils\Documents\Masterarbeit\sample-augment\data"
    #                           r"\KFoldTrainedClassifiers\aug-01_ecc814.json")
    # with open(kfold_trained_path) as kfold_trained_file:
    #     kfold_trained_data = json.load(kfold_trained_file)
    #
    # k_fold_plot_loss_over_epochs(KFoldTrainedClassifiers.from_dict(kfold_trained_data),
    #                              Path(r"C:\Users\Nils\Documents\Masterarbeit\sample-augment\data\shared"),
    #                              "aug-01")
