from typing import Optional

from prototype.step import Step
from prototype.state_store import StateStore


class LoadDataset(Step):
    @staticmethod
    def load_from(self, store: StateStore):
        pass

    @staticmethod
    def check_environment() -> Optional[str]:
        pass
