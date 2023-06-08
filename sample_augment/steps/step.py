from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Type, Dict, Any, Callable, List

from pydantic import BaseModel

from sample_augment.data.artifact import Artifact
from sample_augment.utils import log


# Function to convert snake_case names to CamelCase
def _snake_to_camel(word):
    return ''.join(x.capitalize() or '_' for x in word.split('_'))


class Step(Callable, BaseModel):
    name: str
    func: Callable

    state_args: Dict[str, Type[Artifact]]
    config_args: Dict[str, Type[Any]]

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return self.name + "Step"

    # def __eq__(self, other: Step):
    #     # TODO check what happens when creating two steps with the same name
    #     return self.name == other.name

    def validate_config(self):
        # TODO the config types a step method can receive are either *top-level* config entries or
        #  *subconfig* entries. Before running a step method we should check that the configs are present.
        #  Preferably in the dry_run.
        # entries
        pass


class StepDecorator:
    # TODO document this class well since the code is complex
    # TODO maybe this class can then be removed. Would be good to remove this state
    def __init__(self):
        self.all_steps = {}

    def __repr__(self):
        return f"{str(self.all_steps)}"

    def get_all_steps(self):
        return self.all_steps

    # TODO env_check callable as optional parameter
    def step(self, name=None):
        def decorator(func):
            # Default step_id is the function name converted to CamelCase

            nonlocal name
            if name is None:
                name = _snake_to_camel(func.__name__)

            if name in self.all_steps:
                raise ValueError(
                    f"Step ID '{name}' is already registered. Please choose a different step ID.")

            sig = inspect.signature(func)
            # TODO don't force the user to call this 'state'. instead check each argument if it's a subclass
            #   of StateBundle
            # TODO be more flexible, make accepting a state optional and allow for acceptance of multiple
            #  different states
            # TODO allow returning a tuple of State

            config_kwargs = {}
            state_kwargs = {}
            # input_state_class = sig.parameters['state'].annotation if 'state' in sig.parameters else None
            for param_name, param in sig.parameters.items():
                if issubclass(param.annotation, Artifact):
                    state_kwargs[param_name] = param.annotation
                else:
                    # assert somehow that this is a config
                    config_kwargs[param_name] = param.annotation

            # input_states = []
            # if input_state_class:
            #     input_states.append(input_state_class)

            self.all_steps[name] = Step(
                name=name,
                func=func,
                state_args=state_kwargs,
                config_args=config_kwargs
            )
            log.debug(f'Registered step {name}.')
            return func

        return decorator

    def get_step(self, name) -> Step:
        if name not in self.all_steps:
            raise ValueError(f"Step with name {name} is not registered in StepManager. Available steps: "
                             f"{self.all_steps}")
        return self.all_steps[name]


def import_step_modules(root_modules: List[str]):
    """
    @param root_modules: List of root modules to search for @step methods recursively.
    @return: List of Step instances constructed from found methods decorated with @step.
    """
    for root_module in root_modules:
        # Recursively import all submodules in the package
        package = importlib.import_module(root_module)
        prefix = package.__name__ + "."
        for _importer, modname, _ispkg in pkgutil.walk_packages(package.__path__, prefix):
            _module = importlib.import_module(modname)


# step manager singleton instance for accessing all steps.
_step_decorator = StepDecorator()
# redeclare the step decorator so it's easily importable
step = _step_decorator.step

# TODO needed? remove
get_step = _step_decorator.get_step
