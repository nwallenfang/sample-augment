from pathlib import Path

import click

from sample_augment.core.config import read_config
from sample_augment.core.experiment import Experiment
from sample_augment.core.step import find_steps


# @click.group()
# def cli():
#     # TODO
#     pass


@click.command()
@click.option('arg_config', '--config', default=None, type=click.Path(), help='Path to the configuration '
                                                                              'file.')
def main(arg_config: Path = None):
    """
        CLI for running experiments concerning
    """

    # I could see there some arg parsing going on before constructing a full Config instance.
    # so in the future it won't be like it is now with a whole json file always being parsed
    config = read_config(arg_config)

    find_steps(include=['test', 'data', 'models'], exclude=['models.stylegan2'])

    # create Experiment instance
    experiment = Experiment(config)
    # TODO re-add dry run
    # experiment.dry_run()
    # TODO providing additional artifacts doesn't change the run identifier so this could bring issues
    # evaluate_k_classifiers, k_fold_plot_loss_over_epochs, imagefolder_to_tensors, k_fold_train_classifier
    experiment.run("k_fold_plot_loss_over_epochs")
    # , initial_artifacts=[GC10Folder(
    #                     image_dir=Path(r"C:\Users\Nils\Documents\Masterarbeit\sample-augment\data\raw\gc10-mini"),
    #                     label_dir=Path(r"C:\Users\Nils\Documents\Masterarbeit\sample-augment\data\raw"
    #                                    r"\gc10_labels"))]


# @main.command(name="list")
# def list_steps():
#     print("list")


# @main.command()
def hello():
    click.echo("hello script")


if __name__ == '__main__':
    main()
