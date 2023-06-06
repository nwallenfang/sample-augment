import importlib
import json
import pkgutil
import sys
from json import JSONDecodeError
from pathlib import Path

import click
from pydantic import ValidationError

from sample_augment.prototype.experiment import Experiment
from sample_augment.prototype.params import Params
from sample_augment.prototype.step_id import StepID
from sample_augment.utils.log import log


def init():
    from sample_augment.prototype.step import Step

    # Load all modules in the "steps" package to have ExperimentStep.__subclasses__ return them all
    # needed for StepID validation (step names provided with the config.json for example)
    package_name = 'sample_augment.steps'
    package = importlib.import_module(package_name)

    for _, module_name, _ in pkgutil.iter_modules(package.__path__):
        # Import each module in the package
        importlib.import_module(f'{package_name}.{module_name}')
    all_step_classes = {step_class.__name__: step_class for step_class in Step.__subclasses__()}

    StepID.initialize(possible_ids=list(all_step_classes.keys()))
    Params.step_classes = all_step_classes


@click.command()
def main():
    """
        CLI for running experiments concerning
    """
    init()

    # I could see there some arg parsing going on before constructing a full Config isntance.
    # so in the future it won't be like it is now with a whole json file always being parsed

    config_path = Path('config.json')

    # config_preprocessing
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
        params = Params.parse_obj(param_dict)
    except ValidationError as e:
        print(str(e))
        print(f"Validation failed for {config_path.name}, exiting.")
        sys.exit(1)

    # create Experiment instance
    experiment = Experiment(params)

    experiment.dry_run()
    experiment.run()


# @main.command()
def hello():
    click.echo("hello script")


if __name__ == '__main__':
    main()
