from __future__ import annotations

import sys
from typing import List

from sample_augment.config import Config
from sample_augment.data.state import State, StateBundle
from sample_augment.data.state_store import StateStore, DiskStateStore
from sample_augment.steps.step import Step, OK, MissingConfigEntry, EnvironmentMismatch, \
    MissingDependency, DryRunResult
from sample_augment.utils.log import log


class Experiment:
    """
        TODO class docs
    """
    pipeline: List[Step]
    store: StateStore
    state: State
    config: Config

    def __init__(self, config: Config):
        self.config = config

        # create StateStore instance pointing to directory from config file
        self.store = DiskStateStore(config.root_directory)

        # load the latest state object. If this Experiment has been done before, we will have cached results
        # the state contains the config file
        state = self.store.load_from_config(config)
        self.state = state

        # build Pipeline by instantiating the ExperimentSteps that are declared in the `steps` config entry.
        self.pipeline = []

        for step_id in config.steps:
            self.pipeline.append(Config.step_classes[step_id]())

    def dry_run(self) -> List[DryRunResult]:
        dry_run_state = self.state.copy()  # maybe need to do a deep copy
        errors: List[DryRunResult] = []
        for step in self.pipeline:
            result = step.dry_run(dry_run_state, self.config)
            if isinstance(result, OK):
                # TODO merge states
                dry_run_state = result.state
                continue
            elif isinstance(result, EnvironmentMismatch):
                log.error(f'EnvironmentMismatch in step {step}. Error message: "{result.mismatch_details}"')
            elif isinstance(result, MissingConfigEntry):
                log.error(f'Required config entry missing in step {type(step).__name__}. '
                          f'Missing entries: {result.missing_entries}')
            else:
                assert isinstance(result, MissingDependency)
                log.error(f'Required dependency steps have not run yet for step {step}. '
                          f'Missing dependencies: {result.missing_steps}')

            errors.append(result)

        return errors

    def run(self):
        for step in self.pipeline:
            log.info(f"Running step {type(step).__name__}.")
            # get the StateBundle model this step expects to receive
            # it's a subset of the State
            # and a subclass of StateBundle
            # noinspection PyPep8Naming
            InputStateBundle = step.get_input_state_bundle()

            try:
                state_bundle = self.state.extract_bundle(InputStateBundle)
            except ValueError as err:
                # TODO
                sys.exit(-1)

            output_state: StateBundle = step.run(state_bundle, self.config)

            self.state.completed_steps.append(type(step).__name__)
            self.state.merge_with_bundle(output_state)

        self.store.save(self.state, self.config)
