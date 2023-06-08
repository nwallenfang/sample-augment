import sys
from typing import Any, Union, List, Type

from pydantic import BaseModel

from sample_augment.utils import log


class StateBundle(BaseModel):
    # TODO rename this to something like 'Artifact'. In the new architecture this will be a very central
    #  class and not be that closely coupled to state. Instead State will be more like a collection of
    #  Artifacts
    """
        Represents a subset of the State. Any Step instance expects a StateBundle instance
        in its run() method.
        This base class is basically an empty state.
        Subclasses will extend StateBundle and fill it with the state they need.
    """
    overwrite = True


class State(BaseModel):
    state: dict[str, Any] = {}
    completed_steps: List[str] = []

    def __len__(self):
        return len(self.state)

    def __contains__(self, field: Union[str, Type[StateBundle]]) -> bool:
        if field is str:
            return field in self.state
        else:
            return self._is_satisfiable(type(field))

    def __getitem__(self, item) -> Any:
        return self.state[item]

    def merge_with_bundle(self, bundle: StateBundle):
        if not bundle:
            # skip for empty state
            return
        bundle_type: Type[StateBundle] = type(bundle)

        for field, field_type in bundle_type.__annotations__.items():
            if field not in self.state:
                # easy case, just add the field to state
                self.state[field] = bundle.__getattribute__(field)
            else:
                # already exists, should we overwrite?
                if bundle.overwrite:
                    self.state[field] = bundle.__getattribute__(field)

    # noinspection PyPep8Naming
    def _is_satisfiable(self, SomeStateBundle: Type[StateBundle]) -> bool:
        for field, field_type in SomeStateBundle.__annotations__.items():
            if field not in self.state:
                # log.error(f"Field {field} is missing in State class")
                break
            elif SomeStateBundle.__annotations__[field] != field_type:
                # log.error(f"Field {field} in State class has different type "
                #           f"{SomeStateBundle.__annotations__[field]} than expected {field_type}")
                break
        else:
            if SomeStateBundle.__annotations__:  # else it's empty State that is always satisfiable :)
                # log.error(f"StateBundle {SomeStateBundle.__name__} is unsatisfiable from current State.")
                return False

        return True

    # noinspection PyPep8Naming
    def extract_bundle(self, SomeStateBundle: Type[StateBundle]) -> StateBundle:
        """
            TODO should raise an exception with the msg instead of quitting if it's not satisfiable
        """
        field_dict = {}
        for field, field_type in SomeStateBundle.__annotations__.items():
            if field not in self.state:
                log.error(f"Field {field} is missing in State class")
                break
            elif SomeStateBundle.__annotations__[field] != field_type:
                log.error(f"Field {field} in State class has different type "
                          f"{SomeStateBundle.__annotations__[field]} than expected {field_type}")
                break
            else:
                field_dict[field] = self.state[field]
        else:
            if SomeStateBundle.__annotations__:  # else it's empty State that is always satisfiable :)
                state_bundle: SomeStateBundle = SomeStateBundle.parse_obj(field_dict)
                return state_bundle
        log.error(f"StateBundle {SomeStateBundle.__name__} is unsatisfiable from current State.")
        sys.exit(-1)


