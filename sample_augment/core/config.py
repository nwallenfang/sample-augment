import hashlib
from pathlib import Path

from pydantic import Extra, DirectoryPath, BaseModel, Field, validator


class SubConfig(BaseModel, extra=Extra.allow, allow_mutation=False):
    # might get used for StyleGAN config bundle
    pass


class Config(BaseModel, extra=Extra.allow, allow_mutation=False):
    # Experiment-wide parameters
    name: str
    random_seed: int = 42
    debug = True
    cache = False

    # TODO put root_directory into env instead -> goal: make config.json portable between os/machines
    root_directory: DirectoryPath = Field(exclude=True)

    # train test split params
    train_ratio: float = 0.8,
    min_instances_per_class: int = 10

    target: str = Field(exclude=True)

    # path for files that get saved by steps and are not Artifacts themselves
    figure_directory: Path
    raw_data_directory: Path
    checkpoint_directory: Path

    def get_hash(self):
        json_bytes = self.json(sort_keys=True, exclude={'name': True, 'target': True}).encode('utf-8')
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

    CONFIG_HASH_CUTOFF: int = 5

    @property
    def run_identifier(self):
        return f"{self.get_hash()[:self.CONFIG_HASH_CUTOFF]}"

    @property
    def filename(self):
        return f"{self.name}_{self.get_hash()[:self.CONFIG_HASH_CUTOFF]}"

    @validator('figure_directory', 'raw_data_directory', 'checkpoint_directory', pre=True)
    def assemble_figure_path(cls, v, values):
        if 'root_directory' in values and isinstance(v, str):
            fig_dir: Path = values['root_directory'] / v
            fig_dir.mkdir(exist_ok=True)

            return fig_dir.resolve()
        return v
