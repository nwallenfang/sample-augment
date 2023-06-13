from __future__ import annotations

import sys
from typing import Dict, List

from sample_augment.core import Step, Config, get_step, Store, Artifact
from sample_augment.core.step import step_registry
from sample_augment.utils.log import log


class Experiment:
    """
        TODO class docs
    """
    store: Store
    config: Config

    def __init__(self, config: Config):
        # TODO root_directory through .env file or sys env
        self.config = config

        # load the latest state object. If this Experiment has been done before, we will have cached results
        # the state contains the config file
        # state = self.store.load_from_config(config)
        store_file_path = config.root_directory / f"store_{config.run_identifier}.json"
        if store_file_path.exists():
            self.store = Store.load_from(store_file_path)
            log.info(f"Loading store {store_file_path.name} with artifacts "
                     f"{[a for a in self.store.artifacts]}")

        else:
            self.store = Store(config.root_directory)

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
                sys.exit(-1)

        # TODO type checking, satsifiability
        # extract required entries from config
        try:
            config_args_filled = {key: self.config.__getattribute__(key) for key in step.config_args.keys()}
        except KeyError as err:
            log.error(str(err))
            log.error(f"Entry missing in config")
            sys.exit(-1)
        output_state: Artifact = step(**state_args_filled, **config_args_filled)

        self.store.completed_steps.append(step.name)
        self.store.merge_artifact_into(output_state)

    def run(self, target_name: str, initial_artifacts: List[Artifact] = None):
        # TODO allow for passing a "starting artefact"
        # The ArtifactStore should be located under the root dir and have a name f"store_{config.get_hash()}".
        # if the ArtifactStore already contains the needed Artifact, we can skip the dependency step.
        target = get_step(target_name)
        # TODO ouput smth like either provide this artifact or ???
        pipeline = step_registry.resolve_dependencies(target)
        log.debug(f"{target_name} pipeline: {pipeline}")
        if initial_artifacts:
            pipeline = step_registry.reduce_steps(pipeline, [type(artifact) for artifact in
                                                             initial_artifacts])
            log.debug(f"{target_name} reduced pipeline: {pipeline}")

        if initial_artifacts:
            for artifact in initial_artifacts:
                self.store.merge_artifact_into(artifact)

        for step in pipeline:
            log.info(f"Running step {step.name}.")
            self._run_step(step)

        # Pipeline run is complete, save the produced artifacts and the config that was used
        self.store.save(run_identifier=self.config.run_identifier)

        with open(self.config.root_directory / f"config_{self.config.run_identifier}.json", 'w') as config_f:
            config_f.write(self.config.json(indent=4))

