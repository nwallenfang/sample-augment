import json
import os
import sys
from importlib import import_module
from pathlib import Path
from typing import Dict, List, Union, Type, Any

from sample_augment.core import Artifact
from sample_augment.utils import log


class Store:
    """
        TODO docs
    """
    root_directory: Path
    artifacts: Dict[str, Artifact] = {}
    completed_steps: List[str] = []

    def __init__(self, root_directory: Path):
        self.root_directory = root_directory

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
            self.artifacts[name] = artifact
        elif name in self.artifacts and not overwrite:
            # if not overwrite and this artifact exists already, we'll ignore it
            # (this is not default behavior and I don't know if it has any use)
            log.debug(f'Dropped produced {type(artifact).__name__}.')

    # noinspection PyPep8Naming
    def __geitem__(self, artifact_type: Type[Artifact]) -> Artifact:
        if artifact_type not in self.artifacts:
            raise KeyError(f"Artifact {artifact_type.__name__} is not contained in ArtifactStore.")

        return self.artifacts[artifact_type.__name__]

    def save(self, filename: str, run_identifier: str):
        external_directory = self.root_directory / f"store_{run_identifier}/"
        os.makedirs(self.root_directory, exist_ok=True)
        os.makedirs(external_directory, exist_ok=True)

        data = {}

        for artifact_name, artifact in self.artifacts.items():
            artifact_dict = artifact.serialize(self.root_directory, external_directory)
            if artifact_dict:
                log.debug(f"Saving artifact {artifact_name}")
                data[artifact.fully_qualified_name] = artifact_dict

        store_filename = f'{filename}.json'
        log.info(f"Saving store to {store_filename}")
        with open(self.root_directory / store_filename, 'w') as f:
            try:
                json.dump(data, f, indent=4)
            except TypeError as err:
                log.error(str(err))
                log.error("Couldn't serialize Store, data dict:")
                log.error(data)
                sys.exit(-1)

    @classmethod
    def load_from(cls, store_path: Path):  # TODO maybe pass dependencies instead
        root_directory = store_path.parent  # the dir above the actual store file
        # TODO should this be a class method? hmm
        with open(store_path, 'r') as f:
            data = json.load(f)

        artifacts = {}
        for artifact_name, artifact_data in data.items():
            # TODO test doubly nested module
            # module_name, class_name = artifact_data['__class__'].rsplit('.', 1)
            module_name, class_name = artifact_name.rsplit('.', 1)
            ArtifactSubclass = getattr(import_module(module_name), class_name)
            artifacts[class_name] = ArtifactSubclass.deserialize(artifact_data, root_directory)

        store = cls(root_directory)
        store.artifacts = artifacts
        return store




