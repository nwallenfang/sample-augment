"""
Have different configs/scenarios/runs with results that we want to compare
"""
import json
from typing import List, Dict

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sample_augment.models.evaluate_classifier import KFoldClassificationReport
from sample_augment.models.train_classifier import KFoldTrainedClassifiers, \
    ClassifierMetrics
from sample_augment.utils.path_utils import root_dir
from sample_augment.utils.plot import prepare_latex_plot
import seaborn as sns


def average_f1_per_fold():
    """
        create a plot showing the average best F1 per fold, maybe some folds are easier than others
    """
    figure_directory = root_dir / "shared/classifier_plots"
    kfold_classifier_files = [
        root_dir / "KFoldTrainedClassifiers/aug-01_ecc814.json",
        root_dir / "KFoldTrainedClassifiers/aug-02_f0d3f1.json",
        # root_dir / "KFoldTrainedClassifiers/aug-06-geometric-rework_feaf2f.json",
        root_dir / "KFoldTrainedClassifiers/noaug-04_7d609e.json",
        root_dir / "KFoldTrainedClassifiers/vit-data-aug_325105.json",
    ]
    run_name_overrides = {
        "aug-02": "higher-p ColorJitter"
    }
    classifier_metrics: Dict[str, List[ClassifierMetrics]] = {}

    # collect F1 scores for each fold for each run
    f1_scores = np.empty((len(kfold_classifier_files), 10))  # k-folds = 10
    best_epochs = np.empty((len(kfold_classifier_files), 10))

    for idx_file, kfold_file in enumerate(kfold_classifier_files):
        kfold_json = json.load(open(kfold_file))
        kfold_classifiers: KFoldTrainedClassifiers = KFoldTrainedClassifiers.from_dict(kfold_json)

        metrics = []
        for idx_classifier, classifier in enumerate(kfold_classifiers.classifiers):
            metrics.append(classifier.metrics)

            best_epoch = np.argmax(classifier.metrics.validation_f1)
            best_epochs[idx_file, idx_classifier] = best_epoch
            best_f1 = classifier.metrics.validation_f1[best_epoch]
            f1_scores[idx_file, idx_classifier] = best_f1

        if kfold_classifiers.configs['name'] in run_name_overrides:
            name = run_name_overrides[kfold_classifiers.configs['name']]
        else:
            name = kfold_classifiers.configs['name']

        classifier_metrics[name] = metrics

    folds_index = np.array(range(1, 11))

    # Calculate average F1 score for each fold and sort folds accordingly
    avg_f1_scores_per_fold = np.median(f1_scores, axis=0)
    sorted_fold_indices = np.argsort(avg_f1_scores_per_fold)
    f1_scores = f1_scores[:, sorted_fold_indices]

    prepare_latex_plot()
    plt.figure(figsize=(5, 3))
    for idx, classifier_name in enumerate(classifier_metrics.keys()):
        plt.plot(folds_index, f1_scores[idx, :], marker='o', linestyle='-', label=classifier_name)

    plt.xlabel('Folds ordered by Median F1 Score')
    plt.ylabel('Macro Average F1 Score')
    plt.legend()
    plt.grid(True)
    plt.xticks(folds_index)

    plt.subplots_adjust(left=0.1, bottom=0.15)  # Adjust left and bottom padding
    plt.tight_layout()
    plt.savefig(figure_directory / "f1_over_folds.pdf", bbox_inches="tight")


def class_performance_boxplot():
    k = 10  # number of folds
    figure_directory = root_dir / "shared/classifier_plots"
    kfold_report_files = [
        root_dir / "KFoldClassificationReport/aug-06-geometric-rework_feaf2f.json",
        root_dir / "KFoldClassificationReport/noaug-04_22da73.json",
        root_dir / "KFoldClassificationReport/aug-04_f260d1.json",
        root_dir / "KFoldClassificationReport/vit-data-aug_325105.json",
    ]
    report_names = [
        "aug-06",
        "noaug-04",
        "aug-04",
        "vit-aug"
    ]
    class_name = "rolled_pit"
    _class_idx = 7

    class_precisions = np.empty((len(kfold_report_files), k))
    class_recalls = np.empty((len(kfold_report_files), k))
    class_f1s = np.empty((len(kfold_report_files), k))
    _support = -1

    for idx_file, report_file in enumerate(kfold_report_files):
        reports = KFoldClassificationReport.from_dict(json.load(open(report_file)))

        for idx_fold, report in enumerate(reports.reports):
            class_precisions[idx_file, idx_fold] = report.report[class_name]['precision']
            class_recalls[idx_file, idx_fold] = report.report[class_name]['recall']
            class_f1s[idx_file, idx_fold] = report.report[class_name]['f1-score']
            _support = report.report[class_name]['support']

    prepare_latex_plot()

    plt.figure(figsize=(6, 2.5))
    ax1 = plt.subplot(1, 2, 1)
    df = pd.DataFrame()
    # for each report, add the precision, recall, and f1-score data to the DataFrame
    for i, report_name in enumerate(report_names):
        # we add the report_name to the DataFrame to identify the data
        temp_df = pd.DataFrame({
            'report': report_name,
            'precision': class_precisions[i],
            'recall': class_recalls[i],
            'f1-score': class_f1s[i]
        })
        # append the temporary DataFrame to the main one
        df = df.append(temp_df)

    sns.swarmplot(data=df, x='report', y='precision', hue='report')
    plt.legend([], [], frameon=False)
    plt.ylabel(r'Precision')
    ax1.set_ylim(-0.05, 1.05)
    # plt.yticks([i for i in range(support+1)])
    ax1.set_xlabel('')
    plt.xticks(range(0, len(kfold_report_files)), report_names)
    plt.grid(True)

    ax2 = plt.subplot(1, 2, 2)

    sns.swarmplot(data=df, x='report', y='recall', hue='report')
    plt.legend([], [], frameon=False)
    plt.ylabel(r'Recall')
    ax2.set_ylim(-0.05, 1.05)
    ax2.set_xlabel('')
    plt.xticks(range(0, len(kfold_report_files)), report_names)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(figure_directory / "boxplot.pdf", bbox_inches="tight")


if __name__ == '__main__':
    class_performance_boxplot()
