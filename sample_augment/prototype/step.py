from __future__ import \
    annotations  # TODO will this work in Python 3.7? (or rather: does this have to work in 3.7?)

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from prototype.params import Params
from prototype.state import State, OutputState, InputState

import importlib

from prototype.step_id import StepID


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
class MissingParameter(DryRunResult):
    missing_parameters: List[str]


class Step:
    """
        Class representing a single step in the experiment pipeline.
        This could for example be a step like like Data Normalization, Visualization,
        or Model training.
        TODO: how to support pipelines where one ExperimentStep is run multiple times?
    """
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

    def dry_run(self, state: State, params: Params) -> DryRunResult:
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
            return MissingParameter(missing_parameters)

        # TODO should the state be modified here?
        #   else there is no reason to return the state
        #   maybe change the step_state attribute?
        # TODO change to OutputState instance, would like to make an empty one
        return OK(state)

    @classmethod
    @abstractmethod
    def run(cls, state: InputState, params: Params) -> OutputState:
        pass
