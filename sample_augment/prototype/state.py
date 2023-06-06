from typing import Any

from pydantic import BaseModel


class InputState(BaseModel):
    """
        An instance / subclass of this can be received by any Step in its run() method.
    """
    pass


class OutputState(BaseModel):
    """
        An instance / subclass of this can be returned by any Step in its run() method.
    """
    overwrite_existing = True


class State(BaseModel):
    state: dict[str, Any] = {}

    def __len__(self):
        return len(self.state)

    def merge_with_output(self, output: OutputState):
        # TODO
        pass

    def check_if_contains(self, input: InputState):
       pass