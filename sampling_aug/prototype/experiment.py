from __future__ import annotations

from typing import List

from prototype.config import Config
from prototype.experiment_step import ExperimentStep, OK, MissingConfigEntry, EnvironmentMismatch, \
    MissingDependency, DryRunResult
from prototype.state import State
from prototype.state_store import StateStore
from utils.logging import logger


class Experiment:
    pipeline: List[ExperimentStep]
    state_store: StateStore
    state: State

    def __init__(self, state: State):
        self.state = state

        # build Pipeline by instantiating the ExperimentSteps that are declared in the `steps` config entry.
        self.pipeline = []

        for step_id in state.config.steps:
            self.pipeline.append(Config.step_classes[step_id]())

    def dry_run(self) -> List[DryRunResult]:
        dry_run_state = self.state.copy()  # TODO maybe need to do a deep copy
        errors: List[DryRunResult] = []
        for step in self.pipeline:
            result = step.dry_run(dry_run_state)
            match result:
                case OK(new_state):
                    dry_run_state = new_state
                    continue
                case EnvironmentMismatch(error_message):
                    logger.error(f'EnvironmentMismatch in step {step}. Error message: "{error_message}"')
                case MissingConfigEntry(missing_entries):
                    logger.error(f'Required config entries missing in step {step}. '
                                 f'Missing entries: "{missing_entries}"')
                case MissingDependency(missing_dependencies):
                    logger.error(f'Required dependency steps have not run yet for step {step}. '
                                 f'Missing dependencies: "{missing_dependencies}"')

            errors.append(result)

        return True  # TODO fix

    def run(self):
        for step in self.pipeline:
            step.run(self.state)



