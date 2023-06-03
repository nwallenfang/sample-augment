from abc import abstractmethod
from typing import List


class DataStore:
    pass


class State:
    pass


class StepID:
    """unique identifier for each Step"""
    id: str

    def __init__(self, _id):
        self.id = _id


class Step:
    id: StepID = StepID("abstract-step")
    step_dependencies: List[StepID] = []

    def __init__(self, state_dependency=None, env_dependency=None):
        self.state_dependency = state_dependency
        self.env_dependency = env_dependency

    def run(self, state: State):
        # Run the step's operation
        pass

    @staticmethod
    @abstractmethod
    def load_from(self, store: DataStore):
        # Load state from the store
        pass

    @staticmethod
    @abstractmethod
    def check_environment(self) -> bool:
        # Load state from the store
        pass

    @abstractmethod
    def save_to(self, store):
        # Save state to the store
        pass


class Pipeline:
    # may turn out to be just the list of steps, in which case this class can be removed
    steps: List[Step]


class Config:
    pass


class Experiment:
    def __init__(self, pipeline, data_store: DataStore, config: Config = None):
        self.pipeline = pipeline

    def cold_run(self):
        for step in self.pipeline.steps:
            # Check for step dependencies
            for dep in step.step_dependencies:
                if not dep.load_from(self.pipeline.store):
                    raise Exception(f"Missing step dependency: {dep.name}")

            # Check for state dependency
            if step.state_dependency and not self.pipeline.store.has_state(step.state_dependency):
                raise Exception(f"Missing state dependency: {step.state_dependency}")

            # Check for environment dependency
            if step.env_dependency and not self.check_env(step.env_dependency):
                raise Exception(f"Missing environment dependency: {step.env_dependency}")

            # Run step and save state
            step.run()
            step.save_to(self.pipeline.store)

    def check_env(self, env_dependency):
        # Check if the environment meets the specified dependency
        pass
