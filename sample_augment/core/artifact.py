import hashlib
import inspect
import json
import pprint
import sys
import typing
from importlib import import_module
from pathlib import Path
from typing import *

import numpy as np
import torch
from pydantic import BaseModel, parse_obj_as, ValidationError, Field

from sample_augment.utils import log, path_utils


def is_tuple_of_tensors(field_type):
    if typing.get_origin(field_type) is tuple:
        element_type = typing.get_args(field_type)[0]
        return element_type is torch.Tensor
    return False


class Artifact(BaseModel, arbitrary_types_allowed=True):
    """
        TODO update docs
        Represents a subset of the State. Any Step instance expects a StateBundle instance
        in its run() method.
        This base class is basically an empty state.
        Subclasses will extend StateBundle and fill it with the state they need.
    """
    _serialize_this = True
    config_dependencies: Dict[str, Any] = Field(default_factory=dict)

    @property
    def fully_qualified_name(self):
        return f'{self.__module__}.{self.__class__.__name__}'

    def _serialize_field(self, field, field_name: str, field_type, external_directory: Path):
        # create "run identifier" subdir
        external_directory.mkdir(exist_ok=True)
        filename = f'{self.config_hash}_{field_name}'
        save_path = external_directory / filename
        origin = typing.get_origin(field_type)

        if is_tuple_of_tensors(field_type):
            # TODO we could stack the tensor instead of saving it individually..
            serialized_tensor_strings = []
            # hard-coded specifically for SampleAugmentDataset tensors
            tensors = cast(Tuple[torch.Tensor, ...], field)

            for idx, tensor in enumerate(tensors):
                # serialize each tensor the same way we do it for individual tensors
                filename = f'{self.config_hash}_{field_name}_{idx}.pt'
                save_path = external_directory / filename
                torch.save(tensor, str(save_path))
                serialized_tensor_strings.append({
                    'type': 'torch.Tensor',
                    'path': save_path.relative_to(path_utils.root_directory).as_posix()
                })
            return serialized_tensor_strings
        if not inspect.isclass(field_type):
            # it's a primitive type or a list
            if origin is list or origin is List:  # check if field_type is a list
                # check if it's a list of artifacts
                element_type = get_args(field_type)[0]
                if issubclass(element_type, Artifact):
                    # call upper serialize methdod for Artifacts
                    return [subartifact.serialize() for subartifact in field]
                else:
                    # call serialize_field method for "simple fields"
                    return [
                        self._serialize_field(subfield, f"{field_name}_{i}", type(subfield), external_directory)
                        for i, subfield in enumerate(field)
                    ]

            # primitive type, simply assign
            return field
        # OK at this point it's a class, go through some special cases
        # Tensors and Arrays should get saved to external files (large binary blobs)
        if issubclass(field_type, torch.Tensor):
            save_path = save_path.with_suffix('.pt')
            torch.save(field, str(save_path))
            return {'type': 'torch.Tensor',
                    'path': f'{save_path.relative_to(path_utils.root_directory).as_posix()}'}
        elif issubclass(field_type, torch.nn.Module):
            save_path = save_path.with_suffix('.pt')
            torch.save(field.state_dict(), str(save_path))
            return {'type': 'torch.nn.Module',
                    'class': f'{field.__class__.__module__}.{field.__class__.__name__}',
                    'kwargs': field.get_kwargs(),  # TODO ensure that every model has this
                    'path': f'{save_path.relative_to(path_utils.root_directory).as_posix()}'}
        elif issubclass(field_type, np.ndarray):
            save_path = save_path.with_suffix('.npy')
            np.save(str(save_path), field)
            return {'type': 'numpy.ndarray',
                    'path': f'{save_path.relative_to(path_utils.root_directory).as_posix()}'}
        elif issubclass(field_type, Artifact):
            # field is a sub-artifact. Recursively save it.
            return field.serialize()
        elif issubclass(field_type, Path):
            # make Path instances relative to config.root_dir
            relative_path = field.relative_to(path_utils.root_directory).as_posix()
            return {
                'type': 'pathlib.Path',
                'path': relative_path
            }

        return field

    @staticmethod
    def _deserialize_field(field_name: str, value: Any, type_annotation: Optional[Type[Any]]) -> Any:

        if isinstance(value, dict) and 'path' in value:
            field_path = path_utils.root_directory / value['path']
            if value['type'] == 'torch.Tensor':
                # TODO down the line it could be good/performant to do "lazy loading", so only load once
                #   this artifact is actually being used
                return torch.load(field_path,
                                  map_location=torch.device('cpu'))
            elif value['type'] == 'torch.nn.Module':
                module_name, class_name = value['class'].rsplit('.', 1)
                ModelClass = getattr(import_module(module_name), class_name)
                model = ModelClass(**value['kwargs'])
                model.load_state_dict(
                    torch.load(field_path,
                               map_location=torch.device('cpu')))
                model.eval()
                return model
            elif value['type'] == 'numpy.ndarray':
                return np.load(field_path)
            elif value['type'] == 'pathlib.Path':
                return field_path
            else:  # Artifact type (subartifact)
                module_name, class_name = field_name.rsplit('.', 1)
                ArtifactSubclass = getattr(import_module(module_name), class_name)
                return ArtifactSubclass.load_from(value)
        elif isinstance(value, list):
            # for lists, check if the list values are simple values or key:value fields as well
            first_element = value[0]
            origin = get_origin(type_annotation)
            if origin is list or origin is List:  # checks if type_annotation is a list
                element_type = get_args(type_annotation)[0]  # gets the type of the elements in the list
                if issubclass(element_type, Artifact):  # if list of artifact, special artifact-wise deseres
                    return [element_type.deserialize(subartifact_data) for subartifact_data in value]
            if isinstance(first_element, dict):  # list of complex objects
                # complex object, we need to recursively deserialize each of these
                # the fieldname only gets used in the Artifact branch.
                # TODO check if element is an artifact, else deserialize as it is being done now
                return [Artifact._deserialize_field("anonymous subfield", dict_subvalue, None)
                        for dict_subvalue in value]
            else:
                # list of simple objects, return like this
                return value
        else:
            # simple object, do nothing
            return value

    @property
    def config_hash(self):
        return self._calculate_config_hash(self.config_dependencies)

    @staticmethod
    def _calculate_config_hash(config_dependencies):
        return hashlib.sha256(json.dumps(config_dependencies, sort_keys=True).encode()).hexdigest()[:6]

    # TODO call this method (where?)
    def is_serialized(self):
        external_directory = path_utils.root_directory / self.__class__.__name__
        return (external_directory / f"{self.config_hash}.json").exists()

    def serialize(self) -> Dict:
        if not self._serialize_this:
            return {}

        # config_hash = (hashlib.sha256(json.dumps(self.config_dependencies, sort_keys=True).encode())
        #                .hexdigest()[:6])
        external_directory = path_utils.root_directory / self.__class__.__name__

        data = {}
        for field_name, field_type in self.__annotations__.items():
            field = getattr(self, field_name)

            data[field_name] = self._serialize_field(field, field_name, field_type, external_directory)

        data['config_dependencies'] = self.config_dependencies

        return data

    def save_to_disk(self):
        data = self.serialize()
        external_directory = path_utils.root_directory / self.__class__.__name__
        external_directory.mkdir(exist_ok=True)
        artifact_path = external_directory / f"{self.config_hash}.json"
        log.info(f"Saving artifact to {artifact_path}")
        with open(artifact_path, 'w') as f:
            try:
                json.dump(data, f, indent=4)
            except TypeError as err:
                log.error(str(err))
                log.error("Couldn't serialize Artifact, data dict:")
                log.error(pprint.pformat(data))
                sys.exit(-1)

    @classmethod
    def load_from_disk(cls, config_entries: Dict[str, Any]) -> "Artifact":
        config_hash = cls._calculate_config_hash(config_entries)
        artifact_path = path_utils.root_directory / cls.__name__ / f"{config_hash}.json"
        with open(artifact_path) as artifact_json:
            artifact = cls.deserialize(json.load(artifact_json))
            return artifact

    @classmethod
    def deserialize(cls, data: Dict) -> "Artifact":
        subartifacts = {}

        for field_name, value in data.items():
            # if field_name == 'config_dependencies':
            #     pass
            if field_name not in cls.__fields__:
                # field is not in pydantic model
                continue
            model_field = cls.__fields__[field_name]
            if 'exclude' in model_field.field_info.extra and model_field.field_info.extra['exclude']:
                # field is excluded in pydantic model
                continue
            field_type = model_field.outer_type_  # Get the field's type annotation
            # if field_type is a subclass of Artifact, we need to recursively deserialize it
            # else we can deserialize the field with our helper method.
            if isinstance(field_type, type) and issubclass(field_type, Artifact):
                # save subartifact in external dict and apply after the main loop
                # since the field_name needs to change
                subartifacts[field_name] = field_type.deserialize(value)
            else:
                data[field_name] = Artifact._deserialize_field(field_name, value,
                                                               cls.__annotations__[field_name] if
                                                               field_name in cls.__annotations__ else None)

        for field_name, subartifact in subartifacts.items():
            data.pop(field_name, None)
            data[field_name] = subartifact

        try:
            return parse_obj_as(cls, data)
        except ValidationError as e:
            # failed to validate our deserialized model, something went wrong
            # build a nice error message
            error_messages = e.errors()
            formatted_errors = []

            for error in error_messages:
                path = " -> ".join(str(p) for p in error['loc'])
                # Replace '__root__' with 'MainModel'
                path = path.replace('__root__', cls.__name__)
                msg = error['msg']
                formatted_errors.append(f"  In '{path}', {msg}")

            user_friendly_error_message = "\n".join(formatted_errors)
            log.error(f"Validation errors from deserialization:\n{user_friendly_error_message}")
