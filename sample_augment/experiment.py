from __future__ import annotations

from typing import List

from sample_augment.params import Params
from sample_augment.steps.step import Step, OK, MissingParameter, EnvironmentMismatch, \
    MissingDependency, DryRunResult
from sample_augment.data.state import State, InputState
from sample_augment.data.state_store import StateStore, DiskStateStore
from sample_augment.utils.log import log

import inspect

class Experiment:
    """
        TODO class docs
    """
    pipeline: List[Step]
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

        for step_id in params.steps:
            self.pipeline.append(Params.step_classes[step_id]())

    def dry_run(self) -> List[DryRunResult]:
        dry_run_state = self.state.copy()  # maybe need to do a deep copy
        errors: List[DryRunResult] = []
        for step in self.pipeline:
            result = step.dry_run(dry_run_state, self.params)
            if isinstance(result, OK):
                # TODO merge states
                dry_run_state = result.state
                continue
            elif isinstance(result, EnvironmentMismatch):
                log.error(f'EnvironmentMismatch in step {step}. Error message: "{result.mismatch_details}"')
            elif isinstance(result, MissingParameter):
                log.error(f'Required params missing in step {type(step).__name__}. '
                          f'Missing entries: {result.missing_parameters}')
            else:
                assert isinstance(result, MissingDependency)
                log.error(f'Required dependency steps have not run yet for step {step}. '
                          f'Missing dependencies: {result.missing_steps}')

            errors.append(result)

        return errors

    def run(self):
        for step in self.pipeline:
            ExpectedInputStateClass = inspect.getfullargspec(step.run)
            _output_state = step.run(InputState(), self.params)
            # TODO merge OutputState into whole State
