from __future__ import \
    annotations  # TODO will this work in Python 3.7? (or rather: does this have to work in 3.7?)

from abc import abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from prototype.state import State, ProducedState, ConsumedState

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
class MissingConfigEntry(DryRunResult):
    missing_config_entries: List[str]


class ExperimentStep:
    """
        Class representing a single step in the experiment pipeline.
        This could for example be a step like like Data Normalization, Visualization,
        or Model training.
        TODO: how to support pipelines where one ExperimentStep is run multiple times?
    """
    all_step_ids: List[str] = []
    step_dependencies: List[StepID] = []
    required_configs = []

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

    def dry_run(self, state: State) -> DryRunResult:
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

        missing_config_entries = []
        for required_entry in self.required_configs:
            if required_entry not in state.config:
                missing_config_entries.append(required_entry)
        if missing_config_entries:
            return MissingConfigEntry(missing_config_entries)

        # TODO should the state be modified here?
        #   else there is no reason to return the state
        #   maybe change the step_state attribute?
        return OK(state)

    @classmethod
    @abstractmethod
    def run(cls, state: ConsumedState) -> ProducedState:
        pass
