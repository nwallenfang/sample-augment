import sys
from typing import Any, Union, List, Type

from pydantic import BaseModel

from sample_augment.utils import log


class Artifact(BaseModel):
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


class ArtifactStore(BaseModel):
    artifacts: List[Artifact] = []
    completed_steps: List[str] = []

    def __len__(self):
        return len(self.artifacts)

    def __contains__(self, field: Union[Artifact, Type[Artifact]]) -> bool:
        assert type(field) in [Artifact, Type[Artifact]]

        if isinstance(field, Artifact):
            return field in self.artifacts
        else:
            return self._is_satisfiable(field)

    def __getitem__(self, item) -> Any:
        return self.artifacts[item]

    def merge_artifact_into(self, artifact: Artifact):
        if not artifact:
            # skip for empty state
            return

        # iterate over all fields of the artifact type
        if artifact not in self.artifacts:
            self.artifacts.append(artifact)
        elif artifact.overwrite:
            pass

    # noinspection PyPep8Naming
    def _is_satisfiable(self, SomeArtifact: Type[Artifact]) -> bool:
        for field, field_type in SomeArtifact.__annotations__.items():
            if field not in self.artifacts:
                # log.debug(f"Field {field} is missing in State class")
                return False
            elif SomeArtifact.__annotations__[field] != field_type:
                # log.debug(f"Field {field} in State class has different type "
                #           f"{SomeArtifact.__annotations__[field]} than expected {field_type}")
                return False

        return True

    # noinspection PyPep8Naming
    def extract_bundle(self, SomeArtifact: Type[Artifact]) -> Artifact:
        """
            TODO should raise an exception with the msg instead of quitting if it's not satisfiable
        """
        field_dict = {}
        # for field, field_type in SomeArtifact.__annotations__.items():
        #     if field not in self.artifacts:
        #         log.error(f"Field {field} is missing in State class")
        #         break
        #     elif SomeArtifact.__annotations__[field] != field_type:
        #         log.error(f"Field {field} in State class has different type "
        #                   f"{SomeArtifact.__annotations__[field]} than expected {field_type}")
        #         break
        #     else:
        #         field_dict[field] = self.artifacts[field]
        # else:
        #     if SomeArtifact.__annotations__:  # else it's empty State that is always satisfiable :)
        #         state_bundle: SomeArtifact = SomeArtifact.parse_obj(field_dict)
        #         return state_bundle
        log.error(f"StateBundle {SomeArtifact.__name__} is unsatisfiable from current State.")
        sys.exit(-1)
