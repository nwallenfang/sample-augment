from __future__ import annotations

import json
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Union, Type, Any, Set, Optional

from sample_augment.core import Artifact
from sample_augment.utils import log, path_utils


class Store:
    """
        TODO docs
    """
    artifacts: Dict[str, Artifact] = {}
    completed_steps: List[str] = []

    initial_artifacts: Optional[Set[str]]
    origin_store: Optional[Path]
    # could maybe remove the upper two ones if this works
    previous_run_identifier: Optional[str]

    def __init__(self, artifacts=None):
        # at least look
        if artifacts:
            assert isinstance(artifacts, dict)
            self.artifacts = artifacts

    def __len__(self):
        return len(self.artifacts)

    def __contains__(self, field: Union[Artifact, Type[Artifact]]) -> bool:
        assert type(field) in [Artifact, Type[Artifact]]

        if isinstance(field, Artifact):
            return field in self.artifacts.values()
        else:
            return field.__name__ in self.artifacts.keys()

    def __getitem__(self, item: Type[Artifact]) -> Any:
        return self.artifacts[item.__name__]

    def merge_artifact_into(self, artifact: Artifact, overwrite=True):
        name = type(artifact).__name__

        if not artifact:
            # skip for empty state
            return

        if name not in self.artifacts or overwrite:
            log.debug(f"Added artifact {name}")
            self.artifacts[name] = artifact
        elif name in self.artifacts and not overwrite:
            # if not overwrite and this artifact exists already, we'll ignore it
            # (this is not default behavior and I don't know if it has any use)
            log.debug(f'Dropped produced {type(artifact).__name__}.')

    # def load_artifact(self, artifact_type: Type[Artifact], config_entries: Dict[str, Any]):
    #     # if artifact is not in store currently, we can expect it to be loadable from disk
    #     # since we are checking for that already in the beginning
    #     deserialized_artifact = artifact_type.load_from_disk(config_entries)
    #     self.artifacts[artifact_type.__name__] = deserialized_artifact
    #     return deserialized_artifact

    def save(self, config_name: str, config_hash: str, store_save_path: Optional[Path] = None):
        if store_save_path is None:
            store_save_path = path_utils.root_dir / f"{config_name}_{config_hash}.json"
        elif store_save_path.is_dir():
            store_save_path = store_save_path / f"{config_name}_{config_hash}.json"
        data = {}

        for artifact_name, artifact in self.artifacts.items():
            if artifact is None:
                # FIXME shouldn't be needed
                continue
            if 'name' in artifact.configs and artifact.configs['name'] != config_name:
                log.debug(f"Skipping artifact {artifact_name} since it has another name.")
                continue
            if not artifact.serialize_this:
                continue
            if not artifact.is_serialized(config_name):
                log.warning(f"store.save(): {artifact_name} is not yet serialized! skipping it")
                continue

            # artifact_dict = artifact.serialize()
            # log.debug(f"Saving artifact {artifact_name}")
            data[artifact.__full_name__] = {
                "path": artifact.complete_path.relative_to(path_utils.root_dir).as_posix(),
                "configs": artifact.configs
            }

        if data == {}:
            log.info("Skipped saving empty store.")
            return
        else:
            log.info(f"Store saved: {store_save_path}")
        with open(path_utils.root_dir / store_save_path, 'w') as f:
            try:
                json.dump(data, f, indent=4)
            except TypeError as err:
                log.error(str(err))
                log.error("Couldn't serialize Store, data dict:")
                log.error(data)
                sys.exit(-1)

    @classmethod
    def load_from(cls, store_path: Path):
        # root_directory = store_path.parent  # the dir above the actual store file
        # TODO error handling
        with open(store_path, 'r') as f:
            data = json.load(f)

        artifacts = {}
        for artifact_name, artifact_info in data.items():
            module_name, class_name = artifact_name.rsplit('.', 1)
            artifact_path = path_utils.root_dir / artifact_info['path']
            try:
                ArtifactSubclass = getattr(import_module(module_name), class_name)
            except ModuleNotFoundError as _e:
                log.error(f"Loading from {store_path.name}: Module `{module_name}` does not exist.")
                sys.exit(-1)

            with open(artifact_path, 'r') as artifact_json:
                artifacts[class_name] = ArtifactSubclass.from_dict(json.load(artifact_json))
                if artifacts[class_name] is None:
                    log.warning(f"Error loading {class_name}, skipping it.")
                    del artifacts[class_name]

        store = cls()
        store.artifacts = artifacts
        store.initial_artifacts = set(artifacts)
        store.origin_store = store_path

        # return config if it's present (or move this to Config? not sure)
        # quick and dirty:
        identifier = store_path.name.split('.')[0].split('_')[1]  # fails if name contains a '_'
        # config_path = store_path.parent / f"store_{identifier}" / f"config_{store_path.name}"
        store.previous_run_identifier = identifier
        # stored_config = read_config(config_path)

        return store  # , stored_config

    @staticmethod
    def _get_cached_artifacts(config) -> Dict[str, Artifact]:
        artifacts: Dict[str, Artifact] = {}
        for run_file_name in next(os.walk(path_utils.root_dir))[2]:
            if run_file_name.endswith(".json"):
                run_json = json.load(open(path_utils.root_dir / run_file_name, 'r'))

                for artifact_name, artifact_info in run_json.items():
                    artifact_configs = artifact_info['configs']

                    for config_entry in artifact_configs.keys():
                        if config_entry not in config:
                            log.debug(f"skipping artifact type {artifact_name} since config"
                                      f" {config_entry} is missing")
                            break
                        else:
                            if artifact_configs[config_entry] != getattr(config, config_entry):
                                log.debug(
                                    f"skipping artifact {artifact_name} since its config at {config_entry}"
                                    f"{artifact_configs[config_entry]} != {getattr(config, config_entry)} "
                                    f"(our config)")
                                # TODO there's a bug here, in theory this and above should break and not load the
                                #  artifact. But rn this is the buggy behavior I intend to have so I can more easily
                                #  load cached artifacts. (it's just the name being different which does only matter
                                #  for some artifacts. (unsolved problem!)
                    # artifact is fine and can be loaded
                    module_name, class_name = artifact_name.rsplit('.', 1)
                    # HOTFIX FIXME prefix this with sample_augment if it doesn't contain it
                    if not module_name.startswith('sample_augment.'):
                        module_name = "sample_augment." + module_name
                    ArtifactSubclass = getattr(import_module(module_name), class_name)
                    artifact_dict = json.load(open(path_utils.root_dir / artifact_info['path'], 'r'))
                    artifacts[class_name] = ArtifactSubclass.from_dict(artifact_dict)
                    if artifacts[class_name] is None:
                        log.warning(f"Skipping {class_name}. (error in from_dict)")
                        del artifacts[class_name]
                    log.debug(f"Loaded artifact {class_name} from {run_file_name}.")
        return artifacts

    def merge_cached_artifacts_into(self, config):
        cached_artifacts: Dict[str, Artifact] = Store._get_cached_artifacts(config)
        for cached_name in cached_artifacts.keys():
            if cached_name not in self.artifacts:
                log.debug(f"Adding cached artifact {cached_name} to store.")
                self.artifacts[cached_name] = cached_artifacts[cached_name]

    @staticmethod
    def construct_store_from_cache(config) -> "Store":
        artifacts: Dict[str, Artifact] = {}
        for run_file_name in next(os.walk(path_utils.root_dir))[2]:
            if run_file_name.endswith(".json"):
                run_json = json.load(open(path_utils.root_dir / run_file_name, 'r'))

                for artifact_name, artifact_info in run_json.items():
                    artifact_configs = artifact_info['configs']

                    for config_entry in artifact_configs.keys():
                        if config_entry not in config:
                            log.debug(f"skipping artifact type {artifact_name} since config"
                                      f" {config_entry} is missing")
                            break
                        else:
                            if artifact_configs[config_entry] != getattr(config, config_entry):
                                log.debug(
                                    f"skipping artifact {artifact_name} since its config at {config_entry}"
                                    f"{artifact_configs[config_entry]} != {getattr(config, config_entry)} "
                                    f"(our config)")
                    # artifact is fine and can be loaded
                    module_name, class_name = artifact_name.rsplit('.', 1)
                    try:
                        ArtifactSubclass = getattr(import_module(module_name), class_name)
                    except ModuleNotFoundError as _e:
                        # HOTFIX: try once more by prefixing sample_augment / package name
                        try:
                            ArtifactSubclass = getattr(import_module("sample_augment." + module_name), class_name)
                        except ModuleNotFoundError as e:
                            log.error(e.msg)
                            log.error(
                                f'Could not load artifact {artifact_name}, run-file {run_file_name} might be outdated.')
                            continue
                    artifact_dict = json.load(open(path_utils.root_dir / artifact_info['path'], 'r'))
                    artifacts[class_name] = ArtifactSubclass.from_dict(artifact_dict)
                    if artifacts[class_name] is None:
                        log.warning(f"Error loading {class_name}, skipping it.")
                        del artifacts[class_name]
                    else:
                        log.debug(f"Loaded artifact {class_name} from {run_file_name}.")

        return Store(artifacts=artifacts)
