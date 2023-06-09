import json
import sys
from json import JSONDecodeError
from pathlib import Path

import click
from pydantic import ValidationError

from sample_augment.core.experiment import Experiment
from sample_augment.core.config import Config
from sample_augment.core.step import import_step_modules
from sample_augment.utils.log import log


def read_config(arg_config: Path = None) -> Config:
    if arg_config is None:
        config_path = Path(__file__).parent.parent / 'config.json'
        log.debug(f"Using default config path {config_path.absolute()}")
    else:
        config_path = Path(arg_config)

    # config preprocessing
    try:
        with open(config_path) as json_file:
            param_dict = json.load(json_file)
    except FileNotFoundError as err:
        log.error(str(err))
        sys.exit(-1)
    except JSONDecodeError as err:
        log.error(str(err))
        log.error(f"Failed to parse {config_path.name}, exiting.")
        sys.exit(-1)

    # some preprocessing to have steps conform to StepID format
    param_dict['steps'] = [{'id': x} for x in param_dict['steps']]

    # try reading config file
    try:
        # maybe add command line args to config.json as well
        config = Config.parse_obj(param_dict)
    except ValidationError as e:
        log.error(str(e))
        log.error(f"Validation failed for {config_path.name}, exiting.")
        sys.exit(1)

    return config


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

    import_step_modules(root_modules=['test', 'data'])

    # log.debug(steps)

    # create Experiment instance
    experiment = Experiment(config)

    # TODO re-add dry run
    # experiment.dry_run()
    experiment.run()


# @main.command()
def hello():
    click.echo("hello script")


if __name__ == '__main__':
    main()
