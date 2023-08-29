import json
import os
import sys
from typing import Dict, List

import numpy as np
import seaborn as sns
import torch
from matplotlib import pyplot as plt

from sample_augment.core import Experiment
from sample_augment.core.config import read_config
from sample_augment.core.step import find_steps
from sample_augment.models.evaluate_classifier import KFoldClassificationReport, k_fold_plot_loss_over_epochs
from sample_augment.models.train_classifier import ClassifierMetrics, KFoldTrainedClassifiers
from sample_augment.utils import log
from sample_augment.utils.path_utils import root_dir
from sample_augment.utils.plot import prepare_latex_plot

figures_dir = root_dir.parent / "experiments" / "figures"


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


def check_best_epoch(names, metrics, _reports):
    all_epochs = []

    sns_colors = sns.color_palette("deep", 4)  # Fetch 4 colors from the "deep" palette
    color_groups = {
        sns_colors[0]: ['baseline'],
        sns_colors[1]: ['low-lr', 'high-lr', 'lr-scheduling', 'lr-scheduling-gamma'],
        sns_colors[2]: ['small-batch', 'large-batch'],
        sns_colors[3]: ['color-aug', 'flip-aug', 'geom-aug', 'full-aug']
    }

    color_list = []
    for run_name, run_metrics in zip(names, metrics):
        run_epochs = []

        for fold_metrics in run_metrics:
            # Assuming `fold_metrics` is an object and epoch is an attribute
            best_epoch = fold_metrics.epoch  # Adjust according to your actual data structure
            run_epochs.append(best_epoch)

        for color, group_names in color_groups.items():
            if run_name in group_names:
                color_list.append(color)
                break

        all_epochs.append(run_epochs)

    # Prepare the plot
    prepare_latex_plot()
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.swarmplot(data=all_epochs, ax=ax, palette=color_list)

    ax.set_xticklabels(names)

    plt.xticks(rotation=45)
    plt.ylabel('Epoch of Best Model (max Val. F1)')
    plt.tight_layout()
    plt.savefig(figures_dir / "best_epoch_comparison.pdf", bbox_inches="tight")


def check_macro_f1(names, _metrics, reports):
    all_f1s = []

    sns_colors = sns.color_palette("deep", 4)  # Fetch 4 colors from the "deep" palette
    color_groups = {
        sns_colors[0]: ['baseline'],
        sns_colors[1]: ['low-lr', 'high-lr', 'lr-scheduling', 'lr-scheduling-gamma'],
        sns_colors[2]: ['small-batch', 'large-batch'],
        sns_colors[3]: ['color-aug', 'flip-aug', 'geom-aug', 'full-aug']
    }
    color_list = []

    for run_name, run_reports in zip(names, reports):
        macro_f1_scores = []

        for fold_report in run_reports:
            macro_f1_scores.append(fold_report['macro avg']['f1-score'])
        all_f1s.append(macro_f1_scores)

        # Assign a color to each run_name based on its group
        for color, group_names in color_groups.items():
            if run_name in group_names:
                color_list.append(color)
                break

    prepare_latex_plot()

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.swarmplot(data=all_f1s, ax=ax, palette=color_list)

    baseline_mean_f1 = np.mean(
        [reports[0][i]['macro avg']['f1-score'] for i in range(5)])

    ax.axhline(baseline_mean_f1, color=sns_colors[0], linestyle='--', linewidth=0.9)

    for i, f1_scores in enumerate(all_f1s):
        mean_f1 = np.mean(f1_scores)
        ax.axhline(mean_f1, xmin=(i + 0.3) / len(all_f1s), xmax=(i + 0.7) / len(all_f1s), color='gray')

    ax.set_ylim([0.76, 0.86])
    ax.set_xticklabels(names)
    plt.xticks(rotation=45)
    plt.ylabel('Macro Average F1 Score')

    plt.tight_layout()
    plt.savefig(figures_dir / "f1_comparison.pdf", bbox_inches="tight")


def check_lr_losses(names, metrics, _reports):
    idx_baseline = names.index('baseline')
    # idx_lr_high = names.index('high-lr')
    idx_lr_schedule = names.index('lr-scheduling')

    metrics_baseline = metrics[idx_baseline]
    # metrics_lr_high = metrics[idx_lr_high]
    metrics_lr_schedule = metrics[idx_lr_schedule]

    k_fold_plot_loss_over_epochs(
        {"baseline": metrics_baseline, "lr-schedule": metrics_lr_schedule},
        figures_dir, "lr_comparison"
    )


def subplot_lr_losses(names, metrics, _reports):
    prepare_latex_plot()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    idx_baseline = names.index('baseline')
    idx_lr_schedule = names.index('lr-scheduling')
    idx_full_aug = names.index('full-aug')

    metrics_baseline = metrics[idx_baseline]
    metrics_lr_schedule = metrics[idx_lr_schedule]
    metrics_full_aug = metrics[idx_full_aug]

    yticks = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    ylims = [0.0, 0.6]
    tab10_colors = sns.color_palette("tab10", 10)
    custom_palette = {'baseline': tab10_colors[0], 'lr-schedule': tab10_colors[1], 'full-aug': tab10_colors[2]}

    k_fold_plot_loss_over_epochs({"baseline": metrics_baseline, "lr-schedule": metrics_lr_schedule}, figures_dir,
                                 "lr_comparison", ax=axes[0], yticks=yticks, ylim=ylims, color_dict=custom_palette)
    k_fold_plot_loss_over_epochs({"baseline": metrics_baseline, "full-aug": metrics_full_aug}, figures_dir,
                                 "aug_comparison", ax=axes[1], yticks=yticks, ylim=ylims, color_dict=custom_palette)

    # handles, labels = axes[0].get_legend_handles_labels()
    # fig.legend(handles, labels, loc='upper center', ncol=2)
    plt.tight_layout()
    plt.savefig(figures_dir / "losses_lr_fullaug.pdf", bbox_inches='tight')


def evaluate_classifier_experiment(experiment_name: str):
    experiment_run_dir = root_dir.parent / "experiments" / f"{experiment_name}-runs"
    assert experiment_run_dir.exists(), f"{experiment_run_dir} does not exist."

    _device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # collect experiment artifacts, care this takes tons of RAM like this preloading everything
    results: Dict = {}  # run name to run json
    for run_filename in os.listdir(experiment_run_dir):
        name = run_filename.split("_")[0]
        with open(experiment_run_dir / run_filename) as run_file:
            run_json = json.load(run_file)
            results[name] = run_json

    # TODO include this plot: class_performance_boxplot()
    run_names = sorted(results.keys())
    config = read_config(
        root_dir.parent / "experiments" / f"{experiment_name}-configs" / f"{run_names[0]}.json")
    num_experiments = len(results.keys())
    num_folds = config.n_folds

    metrics = np.empty((num_experiments, num_folds), dtype=object)
    reports = np.empty((num_experiments, num_folds), dtype=object)

    for i, run_name in enumerate(run_names):
        run_data = results[run_name]

        report_path = root_dir / run_data[KFoldClassificationReport.__full_name__]['path']
        kreport = KFoldClassificationReport.from_file(report_path)
        reports[i] = [rep.report for rep in kreport.reports]

        trained_clf_json = json.load(open(root_dir / run_data[KFoldTrainedClassifiers.__full_name__]['path']))
        run_metrics: List[ClassifierMetrics] = [ClassifierMetrics.from_dict(classifier['metrics']) for classifier in
                                                trained_clf_json['classifiers']]
        metrics[i] = run_metrics

    analyses = [
        # check_macro_f1,
        # check_best_epoch,
        # check_lr_losses,
        subplot_lr_losses
    ]

    names = [name[4:] for name in run_names]

    for analysis in analyses:
        analysis(names, metrics, reports)

    # AUC
    # if not torch.cuda.is_available():
    #     log.info('no cuda - no auc metric')
    # predictions = predict_validation_set(classifier, val_set, batch_size=32).predictions
    # auc = roc_auc_score(val_set.label_tensor.numpy(), predictions, average='macro', multi_class='ovr')
    # auc_scores.append(auc)

    # augment_dataset: AugmentDataset = AugmentDataset.from_name('dataset_f00581')
    # bundle = create_train_test_val(augment_dataset, baseline_config.random_seed, baseline_config.test_ratio,
    #                                baseline_config.val_ratio, baseline_config.min_instances)
    # _val_set = bundle.val


def main():
    if len(sys.argv) > 1:
        assert len(sys.argv) == 3, 'run/eval {name}'
        if sys.argv[1] == 'run':
            run_classifier_experiment(sys.argv[2])
        elif sys.argv[1] == 'eval':
            evaluate_classifier_experiment(sys.argv[2])
        else:
            assert False, 'run/eval {name}'
    else:  # default
        run_classifier_experiment("baseline", limit=1)


if __name__ == '__main__':
    main()