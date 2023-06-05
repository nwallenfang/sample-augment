from typing import Optional

from prototype.experiment_step import ExperimentStep
from prototype.state_store import StateStore


class LoadDataset(ExperimentStep):
    @staticmethod
    def load_from(self, store: StateStore):
        pass

    @staticmethod
    def check_environment() -> Optional[str]:
        pass
