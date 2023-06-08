from __future__ import annotations

import inspect
from typing import Type, Dict, Any, Callable

from pydantic import BaseModel

from sample_augment.data.state import StateBundle
from sample_augment.utils import log


# Function to convert snake_case names to CamelCase
def snake_to_camel(word):
    return ''.join(x.capitalize() or '_' for x in word.split('_'))


class Step(Callable, BaseModel):
    name: str
    func: Callable
    input_state_class: Type[StateBundle]
    required_config: Dict[str, Type[Any]]

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return self.name + "Step"


class StepManager:
    # TODO document this class well since the code is complex
    def __init__(self):
        self.all_steps = {}

    def get_all_steps(self):
        return self.all_steps

    # TODO env_check callable as optional parameter
    def step(self, name=None):
        def decorator(func):
            # Default step_id is the function name converted to CamelCase
            nonlocal name
            if name is None:
                name = snake_to_camel(func.__name__)

            if name in self.all_steps:
                raise ValueError(
                    f"Step ID '{name}' is already registered. Please choose a different step ID.")

            sig = inspect.signature(func)
            # TODO don't force the user to call this 'state'. instead check each argument if it's a subclass
            #   of StateBundle
            input_state_class = sig.parameters['state'].annotation
            required_config = {name: param.annotation for name, param in sig.parameters.items() if
                               name != 'state'}

            self.all_steps[name] = Step(
                name=name,
                func=func,
                input_state_class=input_state_class,
                required_config=required_config
            )
            log.debug(f'Registered step {name}.')
            return func

        return decorator

    def get_step(self, name) -> Step:
        if name not in self.all_steps:
            raise ValueError(f"Step with name {name} is not registered in StepManager.")
        return self.all_steps[name]


# step manager singleton instance for accessing all steps.
_step_manager = StepManager()
# redeclare the step decorator so it's easily importable
step = _step_manager.step
get_step = _step_manager.get_step
