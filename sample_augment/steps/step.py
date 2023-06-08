from __future__ import \
    annotations  # TODO will this work in Python 3.7? (or rather: does this have to work in 3.7?)

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List, Optional

from pydantic import BaseModel

from sample_augment.config import Config
from sample_augment.data.state import State, StateBundle

import importlib

from sample_augment.steps.step_id import StepID
from sample_augment.utils import log


# define some rust-like result types, fun :)
@dataclass
class DryRunResult:
    pass


@dataclass
class OK(DryRunResult):
    state: State


@dataclass
class EnvironmentMismatch(DryRunResult):
    mismatch_details: str


@dataclass
class MissingDependency(DryRunResult):
    missing_steps: List[StepID]


@dataclass
class MissingConfigEntry(DryRunResult):
    missing_entries: List[str]


class Step(ABC):
    """
        Class representing a single step in the experiment pipeline.
        This could for example be a step like like Data Normalization, Visualization,
        or Model training.
        TODO: how to support pipelines where one ExperimentStep is run multiple times?
    """
    # TODO read required parameters automatically from run method signature
    #  determine step dependencies automatically from this
    step_dependencies: List[StepID] = []
    required_parameters = []

    def __init__(self, state_dependency=None, env_dependency=None):
        self.state_dependency = state_dependency
        self.env_dependency = env_dependency

    @staticmethod
    def _check_package(package_name: str) -> Optional[str]:
        try:
            importlib.import_module(package_name)
            return None
        except ImportError:
            return f"Package {package_name} is missing in environment."

    @staticmethod
    @abstractmethod
    def check_environment() -> Optional[str]:
        """
            Check if the environment is meeting all requirements to run this step.
            Could for example check the python version or that some package is available.
        """
        raise NotImplementedError()

    @classmethod
    def get_input_state_bundle(cls):
        if cls != Step:
            log.warn(f"{cls.__name__} has no custom InputState (overwrite get_input_state_class())")

        # this method should be overridden by subclasses
        return StateBundle

    def dry_run(self, state: State, params: Config) -> DryRunResult:
        # 1. Check environment
        err = self.check_environment()
        if err:
            return EnvironmentMismatch(err)
        # 2. Check step dependencies
        missing_steps = []
        for dependency_id in self.step_dependencies:
            # TODO is this correct like this?
            #  Additionally it should be checked that the step order is correct.
            if dependency_id not in state:
                missing_steps.append(dependency_id)
        if missing_steps:
            return MissingDependency(missing_steps)

        missing_parameters = []
        for required_entry in self.required_parameters:
            if required_entry not in params:
                missing_parameters.append(required_entry)
        if missing_parameters:
            return MissingConfigEntry(missing_parameters)

        return OK(state)

    @classmethod
    @abstractmethod
    def run(cls, state: StateBundle, params: Config) -> StateBundle:
        pass
