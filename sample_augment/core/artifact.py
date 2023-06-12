import json
import os
from importlib import import_module
from pathlib import Path
from typing import Any, Union, List, Type, Dict

import numpy as np
import torch
from pydantic import BaseModel, parse_obj_as

from sample_augment.utils import log


class Artifact(BaseModel, arbitrary_types_allowed=True):
    """
        TODO update docs
        Represents a subset of the State. Any Step instance expects a StateBundle instance
        in its run() method.
        This base class is basically an empty state.
        Subclasses will extend StateBundle and fill it with the state they need.
    """
    overwrite = False

    @property
    def fully_qualified_name(self):
        return f'{self.__module__}.{self.__class__.__name__}'

    def to_dict(self, root_directory: Path) -> Dict:
        # rename since save is not a fitting name anymore
        # TODO docs
        data = {}

        for field_name, field_type in self.__annotations__.items():
            field = getattr(self, field_name)

            # Tensors and Arrays should get saved to external files (large binary blobs)
            #
            if issubclass(field_type, torch.Tensor):
                save_path = root_directory / f'{self.__class__.__name__}_{field_name}.pt'

                torch.save(field, str(save_path))
                data[field_name] = {'type': 'torch.Tensor',
                                    'path': f'{str(save_path.relative_to(root_directory))}'}
            elif issubclass(field_type, np.ndarray):
                save_path = root_directory / f'{self.__class__.__name__}_{field_name}.npy'
                np.save(str(save_path), field)
                data[field_name] = {'type': 'numpy.ndarray',
                                    'path': f'{str(save_path.relative_to(root_directory))}'}
            elif issubclass(field_type, Artifact):
                # field is a sub-artifact. Recursively save it.
                subartifact_dict = field.to_dict(root_directory)
                data[field_name] = subartifact_dict
            # TODO
            elif issubclass(field_type, Path):
                # make Path instances relative to config.root_dir
                relative_path = field.relative_to(root_directory)
                data[field_name] = {
                    'type': 'pathlib.Path',  # might rather be purepath or something
                    'path': str(relative_path)
                }
            else:
                data[field_name] = field

        return data

    @classmethod
    def from_dict(cls, data: Dict, root_dir: Path):
        # with open(os.path.join(path, f'{cls.__name__}_data.json'), 'r') as f:
        #     data = json.load(f)

        for field, value in data.items():
            if isinstance(value, dict) and 'path' in value:
                if value['type'] == 'torch.Tensor':
                    data[field] = torch.load(os.path.join(root_dir, value['path']))
                elif value['type'] == 'numpy.ndarray':
                    data[field] = np.load(os.path.join(root_dir, value['path']))
                else:  # Artifact type (subartifact)
                    module_name, class_name = field.rsplit('.', 1)
                    ArtifactSubclass = getattr(import_module(module_name), class_name)
                    data[field] = ArtifactSubclass.load_from(value, os.path.join(root_dir))

        return parse_obj_as(cls, data)


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

    def merge_artifact_into(self, artifact: Artifact):
        name = type(artifact).__name__

        if not artifact:
            # skip for empty state
            return

        # iterate over all fields of the artifact type
        if name not in self.artifacts or artifact.overwrite:
            self.artifacts[name] = artifact
        elif name in self.artifacts and not artifact.overwrite:
            log.warning(f'Artifact {type(artifact).__name__} {artifact} getting lost since its type '
                        f'already exists in the Store and overwrite=False.')

    # noinspection PyPep8Naming
    def __geitem__(self, artifact_type: Type[Artifact]) -> Artifact:
        if artifact_type not in self.artifacts:
            raise KeyError(f"Artifact {artifact_type.__name__} is not contained in ArtifactStore.")

        return self.artifacts[artifact_type.__name__]

    def save(self, run_identifier: str):
        os.makedirs(self.root_directory, exist_ok=True)
        data = {}

        for artifact_name, artifact in self.artifacts.items():
            log.info(f"Saving {artifact_name}")
            artifact_dict = artifact.to_dict(self.root_directory)
            data[artifact.fully_qualified_name] = artifact_dict

        with open(self.root_directory / f'store_{run_identifier}.json', 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_from(cls, path: str):  # TODO maybe pass dependencies instead
        # TODO should this be a class method? hmm
        with open(os.path.join(path, 'data.json'), 'r') as f:
            data = json.load(f)

        artifacts = {}
        for artifact_name, artifact_data in data.items():
            # TODO test doubly nested module
            # module_name, class_name = artifact_data['__class__'].rsplit('.', 1)
            module_name, class_name = artifact_name.rsplit('.', 1)
            ArtifactSubclass = getattr(import_module(module_name), class_name)
            artifacts[class_name] = ArtifactSubclass.from_dict(artifact_data, path)

        return cls(artifacts=artifacts)
