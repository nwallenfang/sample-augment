import json
import os
from typing import Dict, List, Optional

import click
import numpy as np

from sample_augment.core import Experiment
from sample_augment.core.config import read_config
from sample_augment.core.step import find_steps
from sample_augment.models.evaluate_baseline import check_macro_f1, subplot_lr_losses, check_best_epoch, \
    check_lr_losses, calc_auc
from sample_augment.models.evaluate_classifier import KFoldClassificationReport
from sample_augment.models.train_classifier import ClassifierMetrics, KFoldTrainedClassifiers
from sample_augment.sampling.evaluate_sampling import sampling_eval
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
    'sampling': 'evaluate_multiseed_synth'
}


def run_experiment(experiment_name, limit=None, run: str = None):
    experiments_dir = root_dir.parent / 'experiments'
    if not running_on_colab():
        find_steps(include=['test', 'data', 'models', 'sampling'], exclude=['models.stylegan2'])
    else:
        log.info("Colab finding steps :)")
        find_steps(
            include=['sample_augment.test', 'sample_augment.data', 'sample_augment.models', 'sample_augment.sampling'],
            exclude=['sample_augment.models.stylegan2']
        )

    # Determine the paths based on the experiment name
    config_path_name = f"{experiment_name}-configs"
    run_path_name = f"{experiment_name}-runs"

    classifier_configs = experiments_dir / config_path_name
    count = 0
    if run:
        run = run + '.json'
        assert (classifier_configs / run).exists()
        run_configs = [run]
    else:
        run_configs = sorted(os.listdir(classifier_configs))

    for config_filename in run_configs:
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


# duplicating this code per experiment because it's faster and easier at the moment
def baseline_eval(results, experiment_name="baseline"):
    # TODO include this plot: class_performance_boxplot() for rolled_pit or other difficult class
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
        subplot_lr_losses,
        calc_auc,  # TODO
    ]

    names = [name[4:] for name in run_names]

    for step in steps:
        step(names, metrics, reports)


def evaluate_experiment(experiment_name: str, run: Optional[str]):
    experiment_run_dir = root_dir.parent / "experiments" / f"{experiment_name}-runs"
    assert experiment_run_dir.exists(), f"{experiment_run_dir} does not exist."

    results: Dict = {}  # run name to run json
    if run:
        run_list = [run]
    else:
        run_list = os.listdir(experiment_run_dir)

    for run_filename in run_list:
        name = run_filename.split("_")[0]
        try:
            with open(experiment_run_dir / run_filename) as run_file:
                run_json = json.load(run_file)
                results[name] = run_json
        except FileNotFoundError as _e:
            log.info(f"Run file {run_filename} not found.")
            continue
    if not results:
        print(run_list)
        log.info("No run files loaded. Passing {run_name: None} as results dict to eval.")
        results = {
            run_file.split("_")[0]: None for run_file in run_list
        }

    if experiment_name == 'baseline':
        baseline_eval(results)
    elif experiment_name == 'sampling':
        sampling_eval(results)
    else:
        raise click.UsageError("No eval implemented for this experiment.")


@click.command()
@click.argument('action', type=click.Choice(['run', 'eval']))
@click.argument('name')
@click.option('--limit', default=None, type=int, help='Limit parameter.')
@click.option('--run', default=None, type=str, help='for running a specific config')
def main(action: str, name: str, limit: Optional[int], run=Optional[str]):
    if action == 'run':
        run_experiment(name, limit, run)
    elif action == 'eval':
        evaluate_experiment(name, run=run)


if __name__ == '__main__':
    main()
