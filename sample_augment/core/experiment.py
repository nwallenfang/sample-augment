from __future__ import annotations

import sys
from typing import *

from sample_augment.core import Step, Config, get_step, Store, Artifact
from sample_augment.core.config import EXCLUDED_CONFIG_KEYS
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

    def __init__(self, config: Config, store: Store = None,
                 load_store: bool = True, save_store: bool = True):
        self.load_store = load_store
        self.save_store = save_store
        self.config = config

        # load the latest state object. If this Experiment has been done before, we will have cached results
        # the state contains the config file
        if store is None:
            store_file_path = root_directory / f"{config.name}_{config.run_identifier}.json"
            if store_file_path.exists() and self.load_store:
                self.store = Store.load_from(store_file_path)
                log.info(f"Continuing with store : {store_file_path.name}")
            else:
                self.store = Store.construct_store_from_cache(config)
        else:
            # when providing an existing store, save a snapshot of the initial artifacts.
            # when saving this store later, only add the new artifacts
            self.store = store
            # self.initial_artifacts = set(store.artifacts)
        log.info(f"Experiment initialized with artifacts: {[a for a in self.store.artifacts]}")

    def __repr__(self):
        return f"Experiment_{self.config.run_identifier}"

    def _run_step(self, step: Step):
        # Get all artifacts from the store that this step receives as arguments
        input_artifacts: Dict[str, Artifact] = {}
        for arg_name, artefact_type in step.consumes.items():
            try:
                artifact = self.store[artefact_type]
                input_artifacts[arg_name] = artifact
            except ValueError as err:
                log.error(str(err))
                sys.exit(-1)
            except KeyError:
                log.error(f"Missing dependency {artefact_type.__name__} while preparing for {step.name}")
                self.save()
                sys.exit(-1)

        # extract required entries from config
        try:
            input_configs = {key: getattr(self.config, key) for key in step.config_args.keys()}
        except (KeyError, AttributeError) as err:
            log.error(str(err))
            log.error(f"Entry missing in config.")
            self.save()
            sys.exit(-1)
        produced: Artifact = step(**input_artifacts, **input_configs)

        # add this step's config args plus all consumed artifact's config args to dependencies
        if produced:
            assert isinstance(produced, Artifact), f"Step {step.name} did not produce an Artifact, but a" \
                                                   f" {type(produced)}"
            # TODO interpret configs more as inputs. Since my assumption that
            produced.configs = {key: value for key, value in input_configs.items() if key not
                                in EXCLUDED_CONFIG_KEYS}
            if 'name' not in produced.configs:
                produced.configs['name'] = self.config.name

            for artifact in input_artifacts.values():
                produced.configs.update(artifact.configs)

            self.store.merge_artifact_into(produced)
            if not produced.is_serialized(self.config.name):
                produced.save_to_disk()

        self.store.completed_steps.append(step.name)

    def run(self, target_name: str, initial_artifacts: List[Artifact] = None):
        target = get_step(target_name)
        pipeline = self._calc_pipeline(initial_artifacts, target)

        for step in pipeline:
            log.info(f"--- {step.name}() ---")
            self._run_step(step)

        if self.save_store:
            # Pipeline run is complete, save the produced artifacts and the config that was used
            self.save()

    def _calc_pipeline(self, initial_artifacts, target):
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
        return pipeline

    def save(self):
        # Pipeline run is complete, save the produced artifacts and the config that was used
        # TODO clean this up, it's a true mess
        if hasattr(self.store, "previous_run_identifier"):
            identifier = self.store.previous_run_identifier
            log.info(f"Using previous store's id {identifier}.")
        else:
            identifier = self.config.run_identifier

        self.store.save(self.config.name, identifier)

        # for now: don't save config file, all the config entries are part of the store file
        # with open(root_directory / f"config_{self.config.name}_{identifier}.json", 'w') as config_f:
        #     config_f.write(self.config.json(indent=4))
