from __future__ import annotations

from abc import abstractmethod
from typing import List

from prototype.config import Config
from prototype.experiment_step import ExperimentStep, DryRunResult, OK, MissingConfigEntry, EnvironmentMismatch, \
    MissingDependency
from prototype.state import State
from utils.logging import logger


class StateStore:
    @abstractmethod
    def load_from_config(self, config: Config) -> State:
        # we'll have to see, maybe we'll only need the config_id (hash)
        pass

    @abstractmethod
    def save(self, state: State):
        pass


class StateDiskStore(StateStore):
    def save(self, state: State):
        raise NotImplementedError()

    def load_from_config(self, config: Config) -> State:
        raise NotImplementedError()


class Experiment:
    pipeline: List[ExperimentStep]
    state_store: StateStore
    state: State

    def __init__(self, pipeline: List[ExperimentStep], state_store: StateStore, config: Config):
        self.pipeline = pipeline
        self.state_store = state_store
        self.state = self.state_store.load_from_config(config)

    def dry_run(self) -> bool:
        dry_run_state = self.state.copy()  # TODO maybe need to do a deep copy

        for step in self.pipeline:
            result = step.dry_run(dry_run_state)
            match result:
                case OK(new_state):
                    dry_run_state = new_state
                case EnvironmentMismatch(error_message):
                    logger.error(f'EnvironmentMismatch in step {step}. Error message: "{error_message}"')
                    return False
                case MissingConfigEntry(missing_entries):
                    logger.error(f'Required config entries missing in step {step}. '
                                 f'Missing entries: "{missing_entries}"')
                    return False
                case MissingDependency(missing_dependencies):
                    logger.error(f'Required dependency steps have not run yet for step {step}. '
                                 f'Missing dependencies: "{missing_dependencies}"')

                    return False

        return True

    def run(self):
        raise NotImplementedError()


"""
        for step in self.pipeline.steps:
            # Check for step dependencies
            for dep in step.step_dependencies:
                if not dep.load_from(self.pipeline.store):
                    raise Exception(f"Missing step dependency: {dep.name}")

            # Check for state dependency
            if step.state_dependency and not self.pipeline.store.has_state(step.state_dependency):
                raise Exception(f"Missing state dependency: {step.state_dependency}")

            # Check for environment dependency
            if step.env_dependency and not self.check_env(step.env_dependency):
                raise Exception(f"Missing environment dependency: {step.env_dependency}")

            # Run step and save state
            step.run()
            step.save_to(self.pipeline.store)
"""
