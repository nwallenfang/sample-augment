import inspect
import typing
from importlib import import_module
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from pydantic import BaseModel, parse_obj_as, ValidationError

from sample_augment.utils import log


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

    @staticmethod
    def _is_tuple_of_tensors(field_type):
        if typing.get_origin(field_type) is tuple:
            element_type = typing.get_args(field_type)[0]
            return element_type is torch.Tensor
        return False

    def _serialize_field(self, field, field_name: str, field_type, root_directory: Path, run_identifier: str):
        # TODO we could maybe reduce some duplications here with some smart method extractions
        save_path_stem = root_directory / run_identifier / f'{self.__class__.__name__}_{field_name}'
        # create "run identifier" subdir
        save_path_stem.parent.mkdir(exist_ok=True)

        if Artifact._is_tuple_of_tensors(field_type):
            serialized_tensor_strings = []
            # hard-coded specifically for SampleAugmentDataset tensors
            tensors = typing.cast(typing.Tuple[torch.Tensor, ...], field)

            for idx, tensor in enumerate(tensors):
                # serialize each tensor the same way we do it for individual tensors
                filename = f'{self.__class__.__name__}_{field_name}_{idx}.pt'
                save_path = save_path_stem.with_name(filename)
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
            save_path = save_path_stem.with_suffix('.pt')
            torch.save(field, str(save_path))
            return {'type': 'torch.Tensor',
                    'path': f'{save_path.relative_to(root_directory).as_posix()}'}
        elif issubclass(field_type, np.ndarray):
            save_path = save_path_stem.with_suffix('.npy')
            np.save(str(save_path), field)
            return {'type': 'numpy.ndarray',
                    'path': f'{save_path.relative_to(root_directory).as_posix()}'}
        elif issubclass(field_type, Artifact):
            # field is a sub-artifact. Recursively save it.
            return field.serialize(root_directory, run_identifier)
        elif issubclass(field_type, Path):
            # make Path instances relative to config.root_dir
            relative_path = field.relative_to(root_directory).as_posix()
            return {
                'type': 'pathlib.Path',
                'path': relative_path
            }

        return field

    @staticmethod
    def _deserialize_field(field_name: str, value: typing.Any, root_dir: Path) -> typing.Any:
        if isinstance(value, dict) and 'path' in value:
            if value['type'] == 'torch.Tensor':
                # TODO down the line it could be good/performant to do "lazy loading", so only load once
                #   this artifact is actually being used
                return torch.load(root_dir / value['path'])
            elif value['type'] == 'numpy.ndarray':
                return np.load(root_dir / value['path'])
            elif value['type'] == 'pathlib.Path':
                return root_dir.joinpath(value['path'])
            else:  # Artifact type (subartifact)
                # assert: see below (list branch)
                assert field_name != "anonymous subfield", "Lists of Artifacts not supported"
                module_name, class_name = field_name.rsplit('.', 1)
                ArtifactSubclass = getattr(import_module(module_name), class_name)
                return ArtifactSubclass.load_from(value, root_dir)
        elif isinstance(value, list):
            # for lists, check if the list values are simple values or key:value fields as well
            first_element = value[0]
            if isinstance(first_element, dict):
                # complex object, we need to recursively deserialize each of these
                # the fieldname only gets used in the Artifact branch. For now we definately don't support
                return [Artifact._deserialize_field("anonymous subfield", dict_subvalue, root_dir)
                        for dict_subvalue in value]
            else:
                # simple object, do nothing
                return value
        else:
            # simple object, do nothing
            return value

    def serialize(self, root_directory: Path, run_identifier: str) -> Dict:
        if not self._serialize_this:
            return {}
        # TODO docs
        data = {}
        for field_name, field_type in self.__annotations__.items():
            field = getattr(self, field_name)

            data[field_name] = self._serialize_field(field, field_name, field_type, root_directory,
                                                     run_identifier)

        return data

    @classmethod
    def deserialize(cls, data: Dict, root_dir: Path):
        # with open(os.path.join(path, f'{cls.__name__}_data.json'), 'r') as f:
        #     data = json.load(f)

        for field_name, value in data.items():
            data[field_name] = Artifact._deserialize_field(field_name, value, root_dir)

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
