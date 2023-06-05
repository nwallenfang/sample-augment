import importlib
import json
import logging
import pkgutil
from json import JSONDecodeError

import click
import sys

from prototype.config import Config
from pydantic import ValidationError

from prototype.experiment import Experiment
from prototype.state_store import DiskStateStore
from utils.logging import logger


def load_step_classes():
    from prototype.experiment_step import ExperimentStep

    # Load all modules in the "steps" package to have ExperimentStep.__subclasses__ return them all
    # needed for StepID validation (step names provided with the config.json for example)
    package_name = 'steps'
    package = importlib.import_module(package_name)

    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        # Import each module in the package
        importlib.import_module(f'{package_name}.{module_name}')
    all_step_classes = {step_class.__name__: step_class for step_class in ExperimentStep.__subclasses__()}
    all_step_ids = list(all_step_classes.keys())

    return all_step_classes, all_step_ids


@click.command()
def main():
    """
        CLI for running experiments concerning
    """
    # TODO needs to be somewhere else
    Config.step_classes, Config.all_step_ids = load_step_classes()

    # I could see there some arg parsing going on before constructing a full Config isntance.
    # so in the future it won't be like it is now with a whole json file always being parsed

    # config_preprocessing
    with open('config.json') as json_file:
        try:
            config_dict = json.load(json_file)
        except JSONDecodeError as err:
            print(err.msg)
            logger.error("Error parsing JSON, exiting.")
            sys.exit(-1)

    # some preprocessing to have steps conform to StepID format
    config_dict['steps'] = [{'id': x} for x in config_dict['steps']]

    # try reading config file
    try:
        # maybe add command line args to config.json as well
        config = Config.parse_obj(config_dict)
    except ValidationError as e:
        print(str(e))
        print("Invalid config file provided, exiting.")
        sys.exit(1)

    print(config)

    # create StateStore instance pointing to directory from config file
    store = DiskStateStore(config.root_directory)

    # load the latest state object. If this Experiment has been done before, we will have cached results
    # the state contains the config file
    state = store.load_from_config(config)

    print(config.steps)
    # create Experiment instance
    experiment = Experiment(state)


# @main.command()
def hello():
    click.echo("hello script")


if __name__ == '__main__':
    main()
