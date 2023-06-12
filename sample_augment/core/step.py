from __future__ import annotations

import importlib
import inspect
import pkgutil
from typing import Type, Dict, Any, Callable, List

from pydantic import BaseModel

from sample_augment.core.artifact import Artifact
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


class StepRegistry:
    # TODO document this class well since the code is complex
    # TODO maybe this class can then be removed. Would be good to remove this state
    all_steps = {}
    producers: Dict[Type[Artifact], List[Step]] = {}
    consumers: Dict[Type[Artifact], List[Step]] = {}

    def __repr__(self):
        return f"{str(self.all_steps)}"

    def get_all_steps(self):
        return self.all_steps

    # TODO env_check callable as optional parameter
    def step(self, func_or_name=None):
        if callable(func_or_name):  # If used as a simple decorator, arg is the decorated function
            return self._register_step(func_or_name)

        else:  # If used as a decorator factory, arg is the argument passed to the factory
            def decorator(func):
                return self._register_step(func, func_or_name)
            return decorator

    def _register_step(self, func, name=None):
        # Default step_id is the function name converted to CamelCase
        if name is None:
            name = _snake_to_camel(func.__name__)

        if name in self.all_steps:
            raise ValueError(
                f"Step ID '{name}' is already registered. Please choose a different step ID.")

        sig = inspect.signature(func)

        config_kwargs = {}
        state_kwargs = {}
        for param_name, param in sig.parameters.items():
            if issubclass(param.annotation, Artifact):
                state_kwargs[param_name] = param.annotation
            else:
                # assert somehow that this is a config
                config_kwargs[param_name] = param.annotation

        new_step = Step(
            name=name,
            func=func,
            state_args=state_kwargs,
            config_args=config_kwargs
        )

        for artifact in new_step.state_args.values():
            self.consumers.setdefault(artifact, []).append(new_step)

        # if this function returns something it should be an Artifact, and it will get added to producers
        if 'return' in func.__annotations__:
            produced_artifact = func.__annotations__['return']
            if not issubclass(produced_artifact, Artifact):
                raise ValueError(
                    f"Return type {produced_artifact} of step '{name}' is not a subclass of Artifact.")
            self.producers.setdefault(produced_artifact, []).append(new_step)

        self.all_steps[name] = new_step
        log.debug(f'Registered step {name}.')
        return func

    def get_step(self, name) -> Step:
        if name not in self.all_steps:
            raise ValueError(f"Step with name {name} is not registered in StepManager. Available steps: "
                             f"{self.all_steps}")
        return self.all_steps[name]

    def resolve_dependencies(self, target_step: Step):
        deps = []
        for artifact in target_step.state_args.values():
            if artifact not in self.producers:
                raise ValueError(f"No step found that produces {artifact}")
            for producer in self.producers[artifact]:
                deps.append(producer)
                deps.extend(self.resolve_dependencies(producer))
        return deps


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
step_registry = StepRegistry()
# redeclare the step decorator so it's easily importable
step = step_registry.step
get_step = step_registry.get_step
