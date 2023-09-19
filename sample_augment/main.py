from pathlib import Path

import click

from sample_augment.core.config import read_config
from sample_augment.core.experiment import Experiment
from sample_augment.core.step import find_steps


@click.command()
@click.option('arg_config', '--config', default=None, type=click.Path(), help='Path to the configuration '
                                                                              'file.')
def main(arg_config: Path = None):
    """
        CLI for running experiments concerning
    """
    config = read_config(arg_config)
    find_steps(include=['test', 'data', 'models', 'sampling'], exclude=['models.stylegan2'])
    # create Experiment instance
    experiment = Experiment(config)

    # evaluate_k_classifiers, k_fold_plot_loss_over_epochs, imagefolder_to_tensors, k_fold_train_classifier
    # experiment.run("train_augmented_classifier")
    # experiment.run("evaluate_classifier")
    # experiment.run('look_at_augmented_train_set')
    experiment.run("synth_bundle_compare_classifiers")


if __name__ == '__main__':
    main()
