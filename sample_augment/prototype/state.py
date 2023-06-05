from typing import Any

from pydantic import BaseModel

from prototype.params import Params
from prototype.step_id import StepID


class ConsumedState(BaseModel):
    """
        Every Step can
    """
    pass


class ProducedState(BaseModel):
    """
        Every Step can
    """
    pass


class State(BaseModel):
    state: dict[str, Any] = {}

    def __len__(self):
        return len(self.state)
