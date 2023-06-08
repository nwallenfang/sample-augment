from __future__ import annotations

from pydantic import BaseModel


class StepID(BaseModel):
    """unique identifier for each Step"""
    id: str
    # _possible_ids: List[str] = None

    def __eq__(self, other: str | StepID):
        if isinstance(other, str):
            return self.id == other
        elif isinstance(other, StepID):
            return self.id == other.id
        else:
            raise ValueError("The argument must be a str or a StepID instance")

    # @validator("id")
    # def validate_step_ids(cls, step_id: str):
    #     if step_id not in cls._possible_ids:
    #         raise ValueError(f"Invalid StepID {step_id} provided.")
    #     return step_id

    def __hash__(self):
        return hash(self.id)

    # @classmethod
    # def initialize(cls, possible_ids: List[str]):
    #     cls._possible_ids = possible_ids


