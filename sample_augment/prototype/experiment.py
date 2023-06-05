from __future__ import annotations

from typing import List

from prototype.params import Params
from prototype.experiment_step import ExperimentStep, OK, MissingConfigEntry, EnvironmentMismatch, \
    MissingDependency, DryRunResult
from prototype.state import State
from prototype.state_store import StateStore, DiskStateStore
from utils.log import log


class Experiment:
    pipeline: List[ExperimentStep]
    store: StateStore
    state: State
    params: Params

    def __init__(self, params: Params):
        self.params = params

        # create StateStore instance pointing to directory from config file
        store = DiskStateStore(params.root_directory)

        # load the latest state object. If this Experiment has been done before, we will have cached results
        # the state contains the config file
        state = store.load_from_config(params)
        self.state = state

        # build Pipeline by instantiating the ExperimentSteps that are declared in the `steps` config entry.
        self.pipeline = []

        for step_id in state.config.steps:
            self.pipeline.append(Params.step_classes[step_id]())

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
                    log.error(f'EnvironmentMismatch in step {step}. Error message: "{error_message}"')
                case MissingConfigEntry(missing_entries):
                    log.error(f'Required params missing in step {type(step).__name__}. '
                              f'Missing entries: {missing_entries}')
                case MissingDependency(missing_dependencies):
                    log.error(f'Required dependency steps have not run yet for step {step}. '
                              f'Missing dependencies: {missing_dependencies}')

            errors.append(result)

        return True  # TODO fix

    def run(self):
        for step in self.pipeline:
            step.run(self.state)
