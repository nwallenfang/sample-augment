import inspect
import json
import os
import typing
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

    @property
    def fully_qualified_name(self):
        return f'{self.__module__}.{self.__class__.__name__}'

    @staticmethod
    def _is_tuple_of_tensors(field_type):
        if typing.get_origin(field_type) is tuple:
            element_type = typing.get_args(field_type)[0]
            return element_type is torch.Tensor
        return False

    def _serialize_field(self, field, field_name, field_type, root_directory):
        # TODO we could maybe reduce some duplications here with some smart method extractions
        if self._is_tuple_of_tensors(field_type):
            serialized_tensor_strings = []
            # hard-coded specifically for SampleAugmentDataset tensors
            tensors = typing.cast(typing.Tuple[torch.Tensor, ...], field)

            for idx, tensor in enumerate(tensors):
                # serialize each tensor the same way we do it for individual tensors
                save_path = root_directory / f'{self.__class__.__name__}_{field_name}{idx}.pt'
                torch.save(tensor, str(save_path))
                serialized_tensor_strings.append({
                    "type": torch.Tensor,
                    "path": Path("").relative_to(root_directory)
                })
            return serialized_tensor_strings
        if not inspect.isclass(field_type):
            # it's a primitive type or a list
            # simply assign
            return field

        # Tensors and Arrays should get saved to external files (large binary blobs)
        if issubclass(field_type, torch.Tensor):
            save_path = root_directory / f'{self.__class__.__name__}_{field_name}.pt'
            torch.save(field, str(save_path))
            return {'type': 'torch.Tensor',
                    'path': f'{str(save_path.relative_to(root_directory))}'}
        elif issubclass(field_type, np.ndarray):
            save_path = root_directory / f'{self.__class__.__name__}_{field_name}.npy'
            np.save(str(save_path), field)
            return {'type': 'numpy.ndarray',
                    'path': f'{str(save_path.relative_to(root_directory))}'}
        elif issubclass(field_type, Artifact):
            # field is a sub-artifact. Recursively save it.
            return field.to_dict(root_directory)
        # TODO
        elif issubclass(field_type, Path):
            # make Path instances relative to config.root_dir
            relative_path = field.relative_to(root_directory)
            return {
                'type': 'pathlib.Path',  # might rather be purepath or something
                'path': str(relative_path)
            }

        return field

    def to_dict(self, root_directory: Path) -> Dict:
        # TODO docs
        data = {}

        for field_name, field_type in self.__annotations__.items():
            field = getattr(self, field_name)
            data[field_name] = self._serialize_field(field, field_name, field_type, root_directory)

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
                elif value['type'] == 'pathlib.Path':
                    data[field] = root_dir.joinpath(value['path'])
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

    def save(self, run_identifier: str):
        os.makedirs(self.root_directory, exist_ok=True)
        data = {}

        for artifact_name, artifact in self.artifacts.items():
            log.debug(f"Saving artifact {artifact_name}")
            artifact_dict = artifact.to_dict(self.root_directory)
            data[artifact.fully_qualified_name] = artifact_dict

        # TODO muy importante save config along with state

        log.info(f"Saving store to store_{run_identifier}.json")
        with open(self.root_directory / f'store_{run_identifier}.json', 'w') as f:
            json.dump(data, f, indent=4)

    @classmethod
    def load_from(cls, path: Path):  # TODO maybe pass dependencies instead
        # TODO should this be a class method? hmm
        with open(path, 'r') as f:
            data = json.load(f)

        artifacts = {}
        for artifact_name, artifact_data in data.items():
            # TODO test doubly nested module
            # module_name, class_name = artifact_data['__class__'].rsplit('.', 1)
            module_name, class_name = artifact_name.rsplit('.', 1)
            ArtifactSubclass = getattr(import_module(module_name), class_name)
            artifacts[class_name] = ArtifactSubclass.from_dict(artifact_data, path)

        store = cls(path.parent)
        store.artifacts = artifacts
        return store
