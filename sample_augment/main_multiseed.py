from pathlib import Path

import click

from sample_augment.core.config import read_config
from sample_augment.core.experiment import Experiment
from sample_augment.core.step import find_steps
from sample_augment.sampling.compare_strategies import MultiSeedStrategyComparison
from sample_augment.utils.path_utils import root_dir
from sample_augment.utils import log


def main():
    """
        CLI for running experiments concerning
    """
    config = read_config(root_dir.parent / 'experiments/sampling-configs/s01-baseline.json')
    find_steps(include=['test', 'data', 'models', 'sampling'], exclude=['models.stylegan2'])
    experiment = Experiment(config)
    log.info('reading MultiSeedStrategyComparison..')
    initial = [MultiSeedStrategyComparison.from_name("s01-baseline_df5f64")]
    log.info('starting step')
    experiment.run("evaluate_multiseed_synth",
                   initial_artifacts=initial)


if __name__ == '__main__':
    main()
