from __future__ import annotations
from pydantic import BaseModel, validator


class StepID(BaseModel):
    """unique identifier for each Step"""
    id: str

    def __eq__(self, other: str | StepID):
        if isinstance(other, str):
            return self.id == other
        elif isinstance(other, StepID):
            return self.id == other.id
        else:
            raise ValueError("The argument must be a str or a StepID instance")

    # TODO I think I can add the validation here again

    def __hash__(self):
        return hash(self.id)