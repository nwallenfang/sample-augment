from typing import Dict, Optional, Union
from pydantic import BaseSettings

class DataPreprocessingConfig(BaseSettings):
    option1: int
    option2: Optional[str]

class ClassifierTrainingConfig(BaseSettings):
    option3: float
    option4: Optional[str]

class Config(BaseSettings):
    learning_rate: float
    num_epochs: int
    steps: Dict[str, Union[DataPreprocessingConfig, ClassifierTrainingConfig]]

    class Config:
        extra = "ignore"
