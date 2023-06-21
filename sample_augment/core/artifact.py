import inspect
import typing
from importlib import import_module
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from pydantic import BaseModel, parse_obj_as, ValidationError

from sample_augment.utils import log, path_utils


def is_tuple_of_tensors(field_type):
    if typing.get_origin(field_type) is tuple:
        element_type = typing.get_args(field_type)[0]
        return element_type is torch.Tensor
    return False


"""
f'{self.__class__.__name__}_{field_name}'
"""


class Artifact(BaseModel, arbitrary_types_allowed=True):
    """
        TODO update docs
        Represents a subset of the State. Any Step instance expects a StateBundle instance
        in its run() method.
        This base class is basically an empty state.
        Subclasses will extend StateBundle and fill it with the state they need.
    """
    _serialize_this = True

    @property
    def fully_qualified_name(self):
        return f'{self.__module__}.{self.__class__.__name__}'

    def _serialize_field(self, field, field_name: str, field_type, root_directory: Path,
                         external_directory: Path):
        # create "run identifier" subdir
        external_directory.parent.mkdir(exist_ok=True)
        filename = f'{self.__class__.__name__}_{field_name}'
        save_path = external_directory / filename

        if is_tuple_of_tensors(field_type):
            # TODO we could stack the tensor instead of saving it individually..
            serialized_tensor_strings = []
            # hard-coded specifically for SampleAugmentDataset tensors
            tensors = typing.cast(typing.Tuple[torch.Tensor, ...], field)

            for idx, tensor in enumerate(tensors):
                # serialize each tensor the same way we do it for individual tensors
                filename = f'{self.__class__.__name__}_{field_name}_{idx}.pt'
                save_path = external_directory / filename
                torch.save(tensor, str(save_path))
                serialized_tensor_strings.append({
                    'type': 'torch.Tensor',
                    'path': save_path.relative_to(root_directory).as_posix()
                })
            return serialized_tensor_strings
        if not inspect.isclass(field_type):
            # it's a primitive type or a list
            # simply assign
            return field
        # OK at this point it's a class, go through some special cases
        # Tensors and Arrays should get saved to external files (large binary blobs)
        if issubclass(field_type, torch.Tensor):
            save_path = save_path.with_suffix('.pt')
            torch.save(field, str(save_path))
            return {'type': 'torch.Tensor',
                    'path': f'{save_path.relative_to(root_directory).as_posix()}'}
        elif issubclass(field_type, torch.nn.Module):
            save_path = save_path.with_suffix('.pt')
            torch.save(field.state_dict(), str(save_path))
            return {'type': 'torch.nn.Module',
                    'class': f'{field.__class__.__module__}.{field.__class__.__name__}',
                    'kwargs': field.get_kwargs(),  # TODO ensure that every model has this
                    'path': f'{save_path.relative_to(root_directory).as_posix()}'}
        elif issubclass(field_type, np.ndarray):
            save_path = save_path.with_suffix('.npy')
            np.save(str(save_path), field)
            return {'type': 'numpy.ndarray',
                    'path': f'{save_path.relative_to(root_directory).as_posix()}'}
        elif issubclass(field_type, Artifact):
            # field is a sub-artifact. Recursively save it.
            return field.serialize(root_directory, external_directory)
        elif issubclass(field_type, Path):
            # make Path instances relative to config.root_dir
            relative_path = field.relative_to(root_directory).as_posix()
            return {
                'type': 'pathlib.Path',
                'path': relative_path
            }

        return field

    @staticmethod
    def _deserialize_field(field_name: str, value: typing.Any) -> typing.Any:

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
                # assert: see below (list branch)
                assert field_name != "anonymous subfield", "Lists of Artifacts not supported"
                module_name, class_name = field_name.rsplit('.', 1)
                ArtifactSubclass = getattr(import_module(module_name), class_name)
                return ArtifactSubclass.load_from(value)
        elif isinstance(value, list):
            # for lists, check if the list values are simple values or key:value fields as well
            first_element = value[0]
            if isinstance(first_element, dict):
                # complex object, we need to recursively deserialize each of these
                # the fieldname only gets used in the Artifact branch. For now, we definately don't support
                return [Artifact._deserialize_field("anonymous subfield", dict_subvalue)
                        for dict_subvalue in value]
            else:
                # simple object, do nothing
                return value
        else:
            # simple object, do nothing
            return value

    def serialize(self, root_directory: Path, external_directory: Path) -> Dict:
        if not self._serialize_this:
            return {}

        data = {}
        for field_name, field_type in self.__annotations__.items():
            field = getattr(self, field_name)

            data[field_name] = self._serialize_field(field, field_name, field_type, root_directory,
                                                     external_directory)

        return data

    @classmethod
    def deserialize(cls, data: Dict, root_dir: Path):
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
                subartifacts[field_name] = field_type.deserialize(value, root_dir)
            else:
                data[field_name] = Artifact._deserialize_field(field_name, value, root_dir)

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
