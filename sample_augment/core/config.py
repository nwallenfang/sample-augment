import hashlib
from typing import Dict, List, ClassVar, Type

from pydantic import Extra, DirectoryPath, BaseModel, Field

from sample_augment.core.step_id import StepID


class SubConfig(BaseModel, extra=Extra.allow, allow_mutation=False):
    pass


class Config(BaseModel, extra=Extra.allow, allow_mutation=False):
    # Experiment-wide parameters
    name: str
    random_seed: int = 42
    debug = True
    cache = False

    # TODO can we make root_dir os independent? hmm doesn't need to be..
    root_directory: DirectoryPath = Field(exclude=True)
    # TODO maybe validate -> StepID
    target: str

    # Step specific settings are saved in the steps dict
    # steps: Dict[str, StepConfig]
    # steps: List[StepID]  # StepID get validated manually

    # bundles = dict[str, SubConfig]

    def get_hash(self):
        json_bytes = self.json(sort_keys=True).encode('utf-8')
        # TODO verify that
        model_hash = hashlib.sha256(json_bytes).hexdigest()

        return model_hash

    @staticmethod
    def create_config_bundles(_cls, values):
        # This dict maps class names to their actual class objects.
        class_name_to_class = {cls.__name__: cls for cls in SubConfig.__subclasses__()}

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
