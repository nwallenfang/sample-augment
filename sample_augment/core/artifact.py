from typing import Any, Union, List, Type, Dict

from pydantic import BaseModel

from sample_augment.utils import log


class Artifact(BaseModel):
    """
        TODO update docs
        Represents a subset of the State. Any Step instance expects a StateBundle instance
        in its run() method.
        This base class is basically an empty state.
        Subclasses will extend StateBundle and fill it with the state they need.
    """
    overwrite = False


class ArtifactStore(BaseModel):
    """
        TODO docs
    """
    artifacts: Dict[str, Artifact] = {}
    completed_steps: List[str] = []

    def __len__(self):
        return len(self.artifacts)

    def __contains__(self, field: Union[Artifact, Type[Artifact]]) -> bool:
        assert type(field) in [Artifact, Type[Artifact]]

        if isinstance(field, Artifact):
            return field in self.artifacts.values()
        else:
            return field.__name__ in self.artifacts.keys()

    def __getitem__(self, item: Type[Artifact]) -> Any:
        return self.artifacts[item.__name__]

    def merge_artifact_into(self, artifact: Artifact):
        name = type(artifact).__name__

        if not artifact:
            # skip for empty state
            return

        # iterate over all fields of the artifact type
        if name not in self.artifacts or artifact.overwrite:
            self.artifacts[name] = artifact
        elif name in self.artifacts and not artifact.overwrite:
            log.warning(f'Artifact {type(artifact).__name__} {artifact} getting lost since its type '
                        f'already exists in the Store and overwrite=False.')

    # noinspection PyPep8Naming
    def __geitem__(self, artifact_type: Type[Artifact]) -> Artifact:
        if artifact_type not in self.artifacts:
            raise KeyError(f"Artifact {artifact_type.__name__} is not contained in ArtifactStore.")

        return self.artifacts[artifact_type.__name__]
