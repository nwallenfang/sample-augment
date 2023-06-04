from typing import Dict, Optional, Union
from pydantic import BaseSettings, Extra


class StepSettings(BaseSettings):
    pass


class EmptyConfigTest(StepSettings):
    pass


class DataPreprocessingConfig(StepSettings):
    option1: int
    option2: Optional[str]


class ClassifierTrainingConfig(StepSettings):
    option3: float
    option4: Optional[str]


# load subclasses of StepSettings, these are the valid keys for StepSettings in the config file.
step_config_classes = {cls.__name__: cls for cls in StepSettings.__subclasses__()}
StepConfig = Union[tuple(step_config_classes.values())]


class Config(BaseSettings, extra=Extra.ignore):
    # Experiment-wide parameters
    dataset_name: str
    random_seed: int = 42
    debug = True
    num_epochs: int
    # Step specific settings are
    steps: Dict[str, Union[DataPreprocessingConfig, ClassifierTrainingConfig]]
