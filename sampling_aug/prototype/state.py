from pydantic import BaseModel

from prototype.config import Config
from prototype.step_id import StepID


class StepState(BaseModel):
    pass


class State(BaseModel):
    config: Config
    step_state: dict[StepID, StepState] = {}

    # def __getitem__(self, step_id: StepID) -> ExperimentStep:
    #     for step in self.step_state:
    #         if step.id == step_id:
    #             return step
    #     else:
    #         raise KeyError(f'invalid key {step_id}, not contained in step_state.')
    #
    # def __contains__(self, step_id: StepID) -> bool:
    #     # if this step
    #     return bool([step for step in self.step_state.keys() if step.id == step_id])

    def __len__(self):
        return len(self.step_state)
