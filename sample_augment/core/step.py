from __future__ import annotations

import functools
import importlib
import inspect
import pkgutil
import pprint
import sys
from typing import Type, Dict, Any, Callable, List, Optional

from pydantic import BaseModel

from sample_augment.core.artifact import Artifact
from sample_augment.utils import log


class Step(Callable, BaseModel):
    """
        Class representing a step in a pipeline. A step is a function with special data containers, Artefacts
        as input and output. Step instances usually get created from normal python functions which are
        decorated with @step.
    """
    name: str
    func: Callable

    """dict (arg_name -> arg_type) of config entries this Step takes as arguments.
       Note that every argument is either an Artifact or a config entry."""
    config_args: Dict[str, Type[Any]]
    """dict (arg_name -> artifact_type) of Artifact types this Step expects as input arguments (consumes)"""
    consumes: Dict[str, Type[Artifact]]
    """Artefact type this step produces (optional)"""
    produces: Optional[Type[Artifact]]

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return self.name

    def __eq__(self, other):
        if isinstance(other, Step):
            return other.name == self.name
        elif callable(other):
            return other == self.func
        else:
            return False

    def __hash__(self):
        # name is unique and immutable
        return hash(self.name)


class StepRegistry:
    # TODO document this class well since the code is complex
    all_steps = {}
    # For more advanced use-cases, it would make sense to have the data structure for the steps be a
    # graph instead of these two dicts
    producers: Dict[Type[Artifact], List[Step]] = {}
    consumers: Dict[Type[Artifact], List[Step]] = {}

    def __repr__(self):
        return f"{str(self.all_steps)}"

    def get_all_steps(self):
        return self.all_steps

    def step(self, name=None):
        if callable(name):  # if used without name argument the "name" is function being decorated
            return self._create_step(name)

        else:
            # if using the name argument, this method @step() is what's called a decorator factory
            # we inject the function name into the decorator and return it
            def decorator(func):
                return self._create_step(func, name)

            return decorator

    @staticmethod
    def _wrap_step(step_instance: Step):
        # the wrapper function to be applied to the original function
        @functools.wraps(step_instance.func)
        def wrapper(*args, **kwargs):
            # TODO this code isn't reached. probably because func in Step gets called and that one isn't
            #  wrapped. We can probably skip that whole wrapping deal and just do our things in experiment
            #  run_Step()
            produced: Artifact = step_instance(*args, **kwargs)
            # Now you can access the step_instance inside the wrapper
            input_artifacts = [arg_value for arg_value in kwargs.values() if isinstance(arg_value, Artifact)]
            input_configs = {arg_name: arg_value for arg_name, arg_value in kwargs.items()
                             if arg_value not in input_artifacts}
            # add this step's config args plus all consumed artifact's config args to dependencies
            produced.config_dependencies = input_configs

            for artifact in input_artifacts:
                produced.config_dependencies.update(artifact.config_dependencies)

            return produced

        return wrapper

    def _create_step(self, func, name=None):
        if name is None:
            name = func.__name__

        if name in self.all_steps:
            # already registered
            return self.all_steps[name]

        sig = inspect.signature(func)

        config_kwargs = {}
        consumed_artifacts = {}
        for param_name, param in sig.parameters.items():
            if issubclass(param.annotation, Artifact):
                consumed_artifacts[param_name] = param.annotation
            else:
                # assert somehow that this is a config
                config_kwargs[param_name] = param.annotation

        produced_artifact = None
        # if this function returns something it should be an Artifact, and it will get added to producers
        if 'return' in func.__annotations__:
            produced_artifact = func.__annotations__['return']
            if not issubclass(produced_artifact, Artifact):
                raise ValueError(
                    f"Return type {produced_artifact} of step '{name}' is not a subclass of Artifact.")

        new_step = Step(
            name=name,
            func=func,
            consumes=consumed_artifacts,
            config_args=config_kwargs,
            produces=produced_artifact
        )

        for artifact in new_step.consumes.values():
            self.consumers.setdefault(artifact, []).append(new_step)

        if produced_artifact:
            self.producers.setdefault(produced_artifact, []).append(new_step)

        self.all_steps[name] = new_step
        log.debug(f'Registered step {name}.')

        return new_step

    def get_step(self, name) -> Step:
        if name not in self.all_steps:
            pretty_list = pprint.pformat(sorted(list(self.all_steps.values()),
                                                key=lambda list_step: list_step.name))
            raise ValueError(f"Step with name {name} is not registered in StepManager. Available steps:\n"
                             f"{pretty_list}")
        return self.all_steps[name]

    def resolve_dependencies(self, target_step: Step) -> List[Step]:
        """
            Create a a pipeline of steps to run.
            TODO document (no cycles, example, produced_artifacts)
        """
        visited = set()
        step_stack = []

        def add_dependencies(node):
            visited.add(node)
            for artifact in node.consumes.values():
                if artifact not in self.producers:
                    raise ValueError(f"No step found that produces {artifact}")

                for producer in self.producers[artifact]:
                    if producer not in visited:
                        add_dependencies(producer)
            step_stack.append(node)

        add_dependencies(target_step)

        return step_stack

    @staticmethod
    def reduce_steps(pipeline: List[Step], initial_artifacts: List[Type[Artifact]]):
        # remove all steps from pipeline whose produced artifacts are contained in initial artifacts
        filtered_pipeline = []
        for pipeline_step in pipeline:
            # for now this expects every step to only produce a single Artifact!
            if pipeline_step.produces in initial_artifacts:
                continue
            else:
                filtered_pipeline.append(pipeline_step)

            # TODO create_train_val_test doesn't get filtered even though it could be
            #  (since its artifact doesn't get serialized)

        return filtered_pipeline


def find_steps(include: List[str], exclude: List[str] = None):
    """
    @param include: List of root modules to search for @step methods recursively.
    @param exclude: List of module names that will be ignored.
    """
    if not exclude:
        exclude = []
    for root_module in include:
        # Recursively import all submodules in the package
        package = importlib.import_module(root_module)
        prefix = package.__name__ + "."
        for _importer, modname, _ispkg in pkgutil.walk_packages(package.__path__, prefix):
            if modname not in sys.modules and not any([mod in modname for mod in exclude]):
                _module = importlib.import_module(modname)


# step manager singleton instance for accessing all steps.
step_registry = StepRegistry()
# redeclare the step decorator so it's easily importable
step = step_registry.step
get_step = step_registry.get_step
