from __future__ import annotations

import sys
from typing import *

from sample_augment.core import Step, Config, get_step, Store, Artifact
from sample_augment.core.step import step_registry, StepRegistry
from sample_augment.utils.log import log
from sample_augment.utils.path_utils import root_directory


class Experiment:
    """
        TODO class docs
    """
    store: Store
    config: Config
    load_store: bool
    save_store: bool

    def __init__(self, config: Config, store: Store = None, load_store: bool = True, save_store: bool = True):
        self.load_store = load_store
        self.save_store = save_store

        self.config = config

        # load the latest state object. If this Experiment has been done before, we will have cached results
        # the state contains the config file
        # state = self.store.load_from_config(config)
        if store is None:
            store_file_path = root_directory / f"{config.name}_{config.run_identifier}.json"
            if store_file_path.exists() and self.load_store:
                self.store, _loaded_config = Store.load_from(store_file_path, root_directory)
                log.info(f"Store has artifacts: {store_file_path.name} "
                         f"{[a for a in self.store.artifacts]}")
            else:
                log.info(f"No existing store for hash {config.run_identifier}. Starting fresh.")
                self.store = Store(root_directory)
        else:
            # when providing an existing store, save a snapshot of the initial artifacts.
            # when saving this store later, only add the new artifacts
            self.store = store
            self.initial_artifacts = set(store.artifacts)

    def __repr__(self):
        return f"Experiment_{self.config.run_identifier}"

    def _run_step(self, step: Step):
        """
            Run passed step, this method doesn't do error checking and
             fails if the store/config doesn't contain necessary artifacts!
        """
        # Get all artifacts from the store that this step receives as arguments
        state_args_filled: Dict[str, Artifact] = {}
        for arg_name, arg_type in step.state_args.items():
            try:
                artifact = self.store[arg_type]
                state_args_filled[arg_name] = artifact
            except ValueError as err:
                log.error(str(err))
                sys.exit(-1)
            except KeyError:
                log.error(f"Missing dependency {arg_type.__name__} while preparing for {step.name}")
                self.save()
                sys.exit(-1)

        # TODO type checking, satsifiability
        # extract required entries from config
        try:
            config_args_filled = {key: self.config.__getattribute__(key) for key in step.config_args.keys()}
        except KeyError as err:
            log.error(str(err))
            log.error(f"Entry missing in config.")
            self.save()
            sys.exit(-1)
        output_state: Artifact = step(**state_args_filled, **config_args_filled)

        self.store.completed_steps.append(step.name)
        self.store.merge_artifact_into(output_state)

    def save(self):
        # Pipeline run is complete, save the produced artifacts and the config that was used
        # TODO clean this up, it's a true mess
        if hasattr(self.store, "previous_run_identifier"):
            identifier = self.store.previous_run_identifier
            log.info(f"Using previous store's id {identifier}.")
        else:
            identifier = self.config.run_identifier

        self.store.save(filename=f"{self.config.name}_{identifier}", run_identifier=identifier)
        external_directory = root_directory / f"store_{identifier}"
        with open(external_directory / f"config_{self.config.name}_{identifier}.json", 'w') as config_f:
            config_f.write(self.config.json(indent=4))

    def run(self, target_name: str, initial_artifacts: List[Artifact] = None):
        target = get_step(target_name)
        full_pipeline = step_registry.resolve_dependencies(target)
        log.debug(f"Pre-Reduce Pipeline: {full_pipeline}")

        if initial_artifacts:
            for artifact in initial_artifacts:
                self.store.merge_artifact_into(artifact)

        # removes the Steps that are not necessary (since their produced Artifacts are already in the Store)
        pipeline = StepRegistry.reduce_steps(full_pipeline, [type(artifact) for artifact in
                                                             self.store.artifacts.values()])
        if target not in pipeline:
            # shouldn't be necessary but doing it because of bug in reduce_steps
            pipeline.append(target)
        if pipeline:
            log.info(f"Running pipeline: {pipeline}")
        else:
            # pipeline empty, so only run the target
            # because doing nothing shouldn't be intended :)
            pipeline.append(target)

        for step in pipeline:
            log.info(f"--- {step.name}() ---")
            self._run_step(step)

        if self.save_store:
            # Pipeline run is complete, save the produced artifacts and the config that was used
            self.save()
