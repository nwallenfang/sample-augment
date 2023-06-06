import hashlib
from pathlib import Path
from typing import Dict, List, ClassVar, Type

from pydantic import BaseSettings, Extra, DirectoryPath, BaseModel

from sample_augment.prototype.step_id import StepID


class ParamBundle(BaseSettings, extra=Extra.allow, allow_mutation=False):
    pass


class Params(BaseModel, extra=Extra.ignore, allow_mutation=False):
    # Experiment-wide parameters
    name: str
    random_seed: int = 42
    debug = True

    root_directory: DirectoryPath
    raw_dataset_path: Path

    # Step specific settings are saved in the steps dict
    # steps: Dict[str, StepConfig]
    steps: List[StepID]  # StepID get validated manually

    """dict from StepID to their step, gets set in main.init()"""
    step_classes: ClassVar[Dict[StepID, Type]]

    bundles = dict[str, ParamBundle]

    def get_hash(self):
        json_bytes = self.json(sort_keys=True).encode('utf-8')
        model_hash = hashlib.sha256(json_bytes).hexdigest()

        return model_hash

    @staticmethod
    def create_config_bundles(_cls, values):
        # This dict maps class names to their actual class objects.
        class_name_to_class = {cls.__name__: cls for cls in ParamBundle.__subclasses__()}

        if 'bundles' in values:
            new_bundles = []
            for bundle_name, bundle_values in values['bundles'].items():
                if bundle_name in class_name_to_class:
                    new_bundles.append(class_name_to_class[bundle_name](**bundle_values))
                else:
                    raise ValueError(f'Unknown bundle type {bundle_name}')
            values['bundles'] = new_bundles
        return values

    def __str__(self):
        return super.__str__(self)

# we could add support for "subtasks", where they are given a new name and associated options
