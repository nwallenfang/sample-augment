from __future__ import annotations

import importlib
import inspect
import pkgutil
import pprint
import sys
from typing import Type, Dict, Any, Callable, List, Optional

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
    produces: Optional[Type[Artifact]]

    # TODO add a method to easily get all consumed/produced Artifacts

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

    def validate_config(self):
        # TODO the config types a step method can receive are either *top-level* config entries or
        #  *subconfig* entries. Before running a step method we should check that the configs are present.
        #  Preferably in the dry_run.
        # entries
        pass


class StepRegistry:
    # TODO document this class well since the code is complex
    all_steps = {}
    producers: Dict[Type[Artifact], List[Step]] = {}
    consumers: Dict[Type[Artifact], List[Step]] = {}

    def __repr__(self):
        return f"{str(self.all_steps)}"

    def get_all_steps(self):
        return self.all_steps

    # TODO env_check callable as optional parameter
    def step(self, name=None):
        if callable(name):  # if used without name argument the "name" is function being decorated
            return self._register_step(name)

        else:
            # if using the name argument, this method @step() is what's called a decorator factory
            # we inject the function name into the decorator and return it
            def decorator(func):
                return self._register_step(func, name)

            return decorator

    def _register_step(self, func, name=None):
        if name is None:
            name = func.__name__  # used to be camel case, but it was confusing

        if name in self.all_steps:
            log.warn(f"Step {name} is already registered.")
            return

        sig = inspect.signature(func)

        config_kwargs = {}
        state_kwargs = {}
        for param_name, param in sig.parameters.items():
            if issubclass(param.annotation, Artifact):
                state_kwargs[param_name] = param.annotation
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
            state_args=state_kwargs,
            config_args=config_kwargs,
            produces=produced_artifact
        )

        # TODO differentiate between "producer" and "transformer" steps
        #   this could be relevant for when we have say a preprocessing step that takes a Dataset and
        #   returns it as well. Don't know if it should really return a new Artifact

        for artifact in new_step.state_args.values():
            self.consumers.setdefault(artifact, []).append(new_step)

        if produced_artifact:
            self.producers.setdefault(produced_artifact, []).append(new_step)

        self.all_steps[name] = new_step
        log.debug(f'Registered step {name}.')
        return func

    def get_step(self, name) -> Step:
        if name not in self.all_steps:
            pretty_list = pprint.pformat(sorted(list(self.all_steps.values()),
                                                key=lambda list_step: list_step.name))
            raise ValueError(f"Step with name {name} is not registered in StepManager. Available steps:\n"
                             f"{pretty_list}")
        return self.all_steps[name]

    def resolve_dependencies(self, target_step: Step) -> List[Step]:
        """
            basically depth-first search topological sort
            TODO document (no cycles, example, produced_artifacts)
        """
        visited = set()
        step_stack = []

        def add_dependencies(node):
            visited.add(node)
            for artifact in node.state_args.values():
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
            # if not pipeline_step.produces:
            #     log.debug(f"skipped step {pipeline_step.name}")
            #     continue  # step not producing anything
            if pipeline_step.produces in initial_artifacts:
                # log.debug(f"Filtered step {pipeline_step.name}")
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
    for root_module in include:
        # Recursively import all submodules in the package
        package = importlib.import_module(root_module)
        prefix = package.__name__ + "."
        for _importer, modname, _ispkg in pkgutil.walk_packages(package.__path__, prefix):
            if modname not in sys.modules and not any([mod in modname for mod in exclude]):
                # log.debug(f"importing {modname}")
                _module = importlib.import_module(modname)


# step manager singleton instance for accessing all steps.
step_registry = StepRegistry()
# redeclare the step decorator so it's easily importable
step = step_registry.step
get_step = step_registry.get_step
