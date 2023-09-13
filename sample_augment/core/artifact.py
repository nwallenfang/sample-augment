import glob
import hashlib
import inspect
import json
import sys
import typing
from importlib import import_module
from pathlib import Path
from typing import *

import numpy as np
import torch
from pydantic import BaseModel, parse_obj_as, ValidationError, Field
# noinspection PyProtectedMember
from pydantic.main import ModelMetaclass

from sample_augment.utils import log, path_utils
from sample_augment.utils.path_utils import root_dir


def is_tuple_of_tensors(field_type):
    if typing.get_origin(field_type) is tuple:
        element_type = typing.get_args(field_type)[0]
        return element_type is torch.Tensor
    return False


class ArtifactMeta(ModelMetaclass):
    """
        due to a problem with producers and consumers in the StepRegistry sometimes having the fully-qualified name
        sometimes not...
    """

    def __init__(cls, name, bases, attrs):
        super().__init__(name, bases, attrs)
        # FIXME sometimes this way of doing it returns 'sample_augment.test...' instead of what we wish to get 'test..'
        #  fixed this in a quick and dirty manner.
        cls.__full_name__ = f"{cls.__module__}.{name}"
        if 'sample_augment' in cls.__full_name__:
            cls.__full_name__ = cls.__full_name__[len('sample_augment') + 1:]


class Artifact(BaseModel, metaclass=ArtifactMeta):
    """
        This is the base class for holding data in the `core` package.
        It supports automatic (de-)serialization of its attributes, including lists, dicts,
        np.arrays, and torch tensors/modules.
        Use Artifact instances to pass inputs and results between `@step` functions.
    """
    """set this to False in your subclass to disable automatic serialization"""
    serialize_this = True
    """dict for tracking the configs this Artifact depends on (by inspecting @step input args)"""
    configs: Dict[str, Any] = Field(default_factory=dict)

    def _serialize_field(self, field, field_name: str, field_type, external_directory: Path,
                         extra_configs: Dict):
        # create "run identifier" subdir
        external_directory.mkdir(exist_ok=True)
        config_hash = self._calculate_config_hash({**self.configs, **extra_configs})
        filename = f'{config_hash}_{field_name}'
        save_path = external_directory / filename
        origin = typing.get_origin(field_type)

        if is_tuple_of_tensors(field_type):
            # TODO we could stack the tensor instead of saving it individually..
            serialized_tensor_strings = []
            # hard-coded specifically for SampleAugmentDataset tensors
            tensors = cast(Tuple[torch.Tensor, ...], field)

            for idx, tensor in enumerate(tensors):
                # serialize each tensor the same way we do it for individual tensors
                filename = f'{config_hash}_{field_name}_{idx}.pt'
                save_path = external_directory / filename
                torch.save(tensor, str(save_path))
                serialized_tensor_strings.append({
                    'type': 'torch.Tensor',
                    'path': save_path.relative_to(path_utils.root_dir).as_posix()
                })
            return serialized_tensor_strings
        if origin is Union:  # only Unions with None (Optional) are supported
            types = get_args(field_type)
            if type(None) in types:
                # It's an Optional type
                actual_type = next(t for t in types if not isinstance(t, type(None)))
                if field is None:
                    return None
                return self._serialize_field(field, field_name, actual_type, external_directory, extra_configs)
        if not inspect.isclass(field_type):
            # it's a primitive type or a list
            if origin is list or origin is List:  # check if field_type is a list
                # check if it's a list of artifacts
                element_type = get_args(field_type)[0]
                if issubclass(element_type, Artifact):
                    # call upper serialize methdod for Artifacts
                    return [subartifact.to_dict() for subartifact in field]
                else:
                    # call serialize_field method for "simple fields"
                    return [
                        self._serialize_field(subfield, f"{field_name}_{i}", type(subfield),
                                              external_directory, extra_configs={
                                **extra_configs, **self.configs
                            })
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
                    'path': f'{save_path.relative_to(path_utils.root_dir).as_posix()}'}
        elif issubclass(field_type, torch.nn.Module):
            save_path = save_path.with_suffix('.pt')
            torch.save(field.state_dict(), str(save_path))
            return {'type': 'torch.nn.Module',
                    'class': f'{field.__class__.__module__}.{field.__class__.__name__}',
                    'kwargs': field.get_kwargs(),  # TODO ensure that every model has this
                    'path': f'{save_path.relative_to(path_utils.root_dir).as_posix()}'}
        elif issubclass(field_type, np.ndarray):
            save_path = save_path.with_suffix('.npy')
            np.save(str(save_path), field)
            return {'type': 'numpy.ndarray',
                    'path': f'{save_path.relative_to(path_utils.root_dir).as_posix()}'}
        elif issubclass(field_type, Artifact):
            # field is a sub-artifact. Recursively save it.
            return field.to_dict(extra_configs={
                **extra_configs, **self.configs
            })
        elif issubclass(field_type, Path):
            # make Path instances relative to config.root_dir
            relative_path = field.relative_to(path_utils.root_dir).as_posix()
            return {
                'type': 'pathlib.Path',
                'path': relative_path
            }

        return field

    @staticmethod
    def _deserialize_field(field_name: str, value: Any, type_annotation: Optional[Type[Any]]) -> Any:
        if isinstance(value, dict) and 'path' in value:
            field_path = path_utils.root_dir / value['path']
            if value['type'] == 'torch.Tensor':
                # TODO down the line it could be good/performant to do "lazy loading", so only load once
                #   this artifact is actually being used
                return torch.load(field_path,
                                  map_location=torch.device('cpu'))
            elif value['type'] == 'torch.nn.Module':
                module_name, class_name = value['class'].rsplit('.', 1)
                ModelClass = getattr(import_module(module_name), class_name)
                model = ModelClass(**value['kwargs'])
                if field_path.is_file():
                    model.load_state_dict(
                        torch.load(field_path,
                                   map_location=torch.device('cpu')))
                else:
                    log.warning(f"Missing model {class_name} at {field_path}")
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
                    return [element_type.from_dict(subartifact_data) for subartifact_data in value]
            if isinstance(first_element, dict):  # list of complex objects
                # complex object, we need to recursively deserialize each of these
                # the fieldname only gets used in the Artifact branch.
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
        return self._calculate_config_hash(self.configs)

    @staticmethod
    def _calculate_config_hash(configs: Dict):
        keys_to_exclude = {'shared_directory', 'transform'}
        filtered_configs = {k: v for k, v in configs.items() if k not in keys_to_exclude}
        return hashlib.sha256(json.dumps(filtered_configs, sort_keys=True).encode()).hexdigest()[:6]

    def is_serialized(self, name: str):
        self.configs['name'] = name
        if self.complete_path.exists():
            return True

        return self.complete_path.exists()

    @classmethod
    def exists(cls, name: str) -> Optional[str]:
        artifact_dir = path_utils.root_dir / cls.__name__
        matching_files = glob.glob(f"{artifact_dir}/{name}_*.json", recursive=True)
        full_names = [Path(f).stem for f in matching_files]

        if len(full_names) == 0:
            return None
        if len(full_names) > 1:
            log.warning(f"Multiple artifacts match the name {name}. Returning the first one.")
        return full_names[0]

    @property
    def complete_path(self) -> Path:
        artifact_dir = path_utils.root_dir / self.__class__.__name__
        if 'name' in self.configs:
            return artifact_dir / f"{self.configs['name']}_{self.config_hash}.json"
        else:
            return artifact_dir / f"noname_{self.config_hash}.json"

    @staticmethod
    def collect_annotations(cls):
        """
        Collects and returns all the field annotations from the given class and its parent classes recursively.
        Used to iterate over all Artifact fields.

        Returns:
            dict: A dictionary mapping field names to their types, including fields from the input class and all its
             parent classes.
        """
        annotations = {}
        if issubclass(cls, BaseModel):
            for base in cls.__bases__:
                annotations.update(Artifact.collect_annotations(base))
            annotations.update(cls.__annotations__)
        return annotations

    def to_dict(self, extra_configs: Dict = None) -> Dict:
        if extra_configs is None:
            extra_configs = {}

        external_directory = path_utils.root_dir / self.__class__.__name__

        data = {}
        # Go through the inheritance chain to collect all fields of itself and parent classes
        # noinspection PyTypeChecker
        all_annotations = Artifact.collect_annotations(self.__class__)
        for field_name, field_type in all_annotations.items():
            field = getattr(self, field_name)

            data[field_name] = self._serialize_field(field, field_name, field_type, external_directory,
                                                     extra_configs=extra_configs)

        data['configs'] = self.configs

        return data

    def save_to_disk(self):
        if not self.serialize_this:
            # FIXME if save_to_disk gets called explicitly it should be fine and not returned
            return
        data = self.to_dict(extra_configs={})
        external_directory = path_utils.root_dir / self.__class__.__name__
        # self.complete_path belongs to external directory. Create it if it doesn't exist yet
        external_directory.mkdir(exist_ok=True)

        log.info(f"Saving artifact to {self.complete_path}")
        with open(self.complete_path, 'w') as f:
            try:
                json.dump(data, f, indent=4)
            except TypeError as err:
                log.error(str(err))
                log.error(f"Couldn't serialize Artifact {self.__full_name__}")
                sys.exit(-1)

    @classmethod
    def from_dict(cls, data: Dict) -> "Artifact":
        subartifacts = {}

        for field_name, value in data.items():
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
                subartifacts[field_name] = field_type.from_dict(value)
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

    @classmethod
    def from_file(cls, path: Union[str, Path]) -> "Artifact":
        with open(path) as json_file:
            return cls.from_dict(json.load(json_file))

    @classmethod
    def from_name(cls, name: str) -> "Artifact":
        # TODO ugly dependency on root dir, hmm
        if not name.endswith(".json"):
            name = name + ".json"
        json_path = root_dir / cls.__name__ / name
        if not json_path.exists():
            log.error(f"{cls.__name__} with filename {name} can't be found.")
        with open(json_path) as json_file:
            return cls.from_dict(json.load(json_file))

    class Config:
        arbitrary_types_allowed = True
