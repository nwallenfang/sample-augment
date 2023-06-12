from __future__ import annotations

import sys
from typing import Dict

from sample_augment.core import Step, Config, get_step, Store, Artifact
from sample_augment.core.step import step_registry
from sample_augment.utils.log import log


class Experiment:
    """
        TODO class docs
    """
    store: Store
    config: Config
    CONFIG_HASH_CUTOFF: int = 5


    def __init__(self, config: Config):
        self.config = config

        # load the latest state object. If this Experiment has been done before, we will have cached results
        # the state contains the config file
        # state = self.store.load_from_config(config)
        self.store = Store(root_directory=config.root_directory)

    def __repr__(self):
        return f"Experiment_{self.config.name}_{self.config.get_hash()[:3]}"

    def run_step(self, step: Step):
        state_args_filled: Dict[str, Artifact] = {}
        for arg_name, arg_type in step.state_args.items():
            try:
                artifact = self.store[arg_type]
                state_args_filled[arg_name] = artifact
            except ValueError as err:
                log.error(str(err))
                sys.exit(-1)

        # TODO type checking, satsifiability
        # extract required entries from config
        config_args_filled = {key: self.config.__getattribute__(key) for key in step.config_args.keys()}
        output_state: Artifact = step(**state_args_filled, **config_args_filled)

        self.store.completed_steps.append(step.name)
        self.store.merge_artifact_into(output_state)

    def run(self, target_name: str):
        # TODO load ArtifactStore.
        # The ArtifactStore should be located under the root dir and have a name f"store_{config.get_hash()}".
        # if the ArtifactStore already contains the needed Artifact, we can skip the dependency step.

        target = get_step(target_name)
        dependencies = step_registry.resolve_dependencies(target)
        for step in dependencies:
            log.info(f"Running dependency {step.name}.")
            # get the StateBundle model this step expects to receive
            # it's a subset of the State
            # and a subclass of StateBundle
            # noinspection PyPep8Naming
            self.run_step(step)

        # run target
        log.info(f"Running target {target.name}.")
        self.run_step(target)

        self.store.save(self.config.get_hash()[:self.CONFIG_HASH_CUTOFF])
