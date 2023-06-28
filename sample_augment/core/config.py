import hashlib
import json
import sys
from json import JSONDecodeError
from pathlib import Path
from typing import Set

from pydantic import Extra, BaseModel, validator, ValidationError

from sample_augment.utils import log, path_utils


class SubConfig(BaseModel, extra=Extra.allow, allow_mutation=False):
    # might get used for StyleGAN config bundle
    pass


EXCLUDED_CONFIG_KEYS: Set[str] = {"shared_directory"}
CONFIG_HASH_CUTOFF: int = 5


class Config(BaseModel, extra=Extra.allow, allow_mutation=False):
    # Experiment-wide parameters
    name: str = "test"

    # path for files that get saved by steps and are not Artifacts themselves
    shared_directory: Path = path_utils.root_directory / "shared"

    # checkpoint_directory: Path = Path("./checkpoints")

    def get_hash(self):
        json_bytes = self.json(sort_keys=True,
                               exclude={entry: True for entry in EXCLUDED_CONFIG_KEYS}).encode('utf-8')
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

    @property
    def run_identifier(self):
        return f"{self.get_hash()[:CONFIG_HASH_CUTOFF]}"

    @property
    def filename(self):
        return f"{self.name}_{self.get_hash()[:CONFIG_HASH_CUTOFF]}"

    @validator('shared_directory', pre=True)
    def assemble_figure_path(cls, value):
        fig_dir: Path = path_utils.root_directory / value

        fig_dir.mkdir(exist_ok=True)
        assert fig_dir.exists()
        return fig_dir.resolve()

    def __contains__(self, item):
        return item in self.__dict__


def read_config(arg_config: Path = None) -> Config:
    if arg_config is None:
        config_path = Path(__file__).parent.parent.parent / 'config.json'
        log.debug(f"Using default config path {config_path.absolute()}")
    else:
        config_path = Path(arg_config)

    # config preprocessing
    try:
        with open(config_path) as json_file:
            param_dict = json.load(json_file)
    except FileNotFoundError as err:
        log.error(str(err))
        sys.exit(-1)
    except JSONDecodeError as err:
        log.error(str(err))
        log.error(f"Failed to parse {config_path.name}, exiting.")
        sys.exit(-1)

    # try reading config file
    try:
        # maybe add command line args to config.json as well
        config = Config.parse_obj(param_dict)
    except ValidationError as e:
        log.error(str(e))
        log.error(f"Validation failed for {config_path.name}, exiting.")
        sys.exit(1)

    return config
