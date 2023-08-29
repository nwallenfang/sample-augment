import json
import os
import sys
from typing import Dict, List

import numpy as np

from sample_augment.core import Experiment
from sample_augment.core.config import read_config
from sample_augment.core.step import find_steps
from sample_augment.models.evaluate_baseline import check_macro_f1, subplot_lr_losses, check_best_epoch, \
    check_lr_losses
from sample_augment.models.evaluate_classifier import KFoldClassificationReport
from sample_augment.models.train_classifier import ClassifierMetrics, KFoldTrainedClassifiers
from sample_augment.utils import log
from sample_augment.utils.path_utils import root_dir


def running_on_colab():
    try:
        import google.colab
        return True
    except ImportError:
        return False


experiment_to_step = {
    'baseline': 'evaluate_k_classifiers',
    'architecture': 'evaluate_k_classifiers',
    'sampling': 'synth_bundle_compare_classifiers_multi_seed'
}


def run_experiment(experiment_name, limit=None):
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


def evaluate_experiment(experiment_name: str):
    experiment_run_dir = root_dir.parent / "experiments" / f"{experiment_name}-runs"
    assert experiment_run_dir.exists(), f"{experiment_run_dir} does not exist."

    results: Dict = {}  # run name to run json
    for run_filename in os.listdir(experiment_run_dir):
        name = run_filename.split("_")[0]
        with open(experiment_run_dir / run_filename) as run_file:
            run_json = json.load(run_file)
            results[name] = run_json

    # TODO include this plot: class_performance_boxplot() for rolled_pit or other difficult class
    # TODO AUC step, for this the TrainedClassifierModels are needed
    run_names = sorted(results.keys())
    config = read_config(
        root_dir.parent / "experiments" / f"{experiment_name}-configs" / f"{run_names[0]}.json")
    num_experiments = len(results.keys())
    num_folds = config.n_folds

    metrics = np.empty((num_experiments, num_folds), dtype=object)
    reports = np.empty((num_experiments, num_folds), dtype=object)

    # load in all metrics (not the TrainedClassifier models themselves!) and classification reports
    for i, run_name in enumerate(run_names):
        run_data = results[run_name]

        report_path = root_dir / run_data[KFoldClassificationReport.__full_name__]['path']
        kreport = KFoldClassificationReport.from_file(report_path)
        reports[i] = [rep.report for rep in kreport.reports]

        trained_clf_json = json.load(open(root_dir / run_data[KFoldTrainedClassifiers.__full_name__]['path']))
        run_metrics: List[ClassifierMetrics] = [ClassifierMetrics.from_dict(classifier['metrics']) for classifier in
                                                trained_clf_json['classifiers']]
        metrics[i] = run_metrics

    steps = [
        check_macro_f1,
        check_best_epoch,
        check_lr_losses,
        subplot_lr_losses
    ]

    names = [name[4:] for name in run_names]

    for step in steps:
        step(names, metrics, reports)

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
            run_experiment(sys.argv[2])
        elif sys.argv[1] == 'eval':
            evaluate_experiment(sys.argv[2])
        else:
            assert False, 'run/eval {name}'
    else:  # default
        run_experiment("baseline", limit=1)


if __name__ == '__main__':
    main()
