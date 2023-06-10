import numpy as np
import torch

from sample_augment.core.artifact import Artifact, Store


class SimpleArtifact(Artifact):
    x: int
    y: float
    z: str


class ComplexArtifact(Artifact):
    simple: SimpleArtifact
    tensor: torch.Tensor
    array: np.ndarray


def test_artifact_store(tmpdir):
    # First, let's create some Artifacts
    simple_artifact = SimpleArtifact(x=1, y=2.0, z='three')
    complex_artifact = ComplexArtifact(simple=simple_artifact,
                                       tensor=torch.tensor([1, 2, 3]),
                                       array=np.array([4, 5, 6]))

    # Then, create a Store from these artifacts and save it
    artifact_store = Store(artifacts={'SimpleArtifact': simple_artifact, 'ComplexArtifact':
                           complex_artifact})
    artifact_store.save(tmpdir)

    # Finally, load the ArtifactStore
    loaded_artifact_store = Store.load_from(tmpdir)

    # Check that the loaded artifacts match the originals
    assert loaded_artifact_store.artifacts['SimpleArtifact'] == simple_artifact
    assert torch.all(loaded_artifact_store.artifacts['ComplexArtifact'].tensor.eq(complex_artifact.tensor))
    assert np.all(loaded_artifact_store.artifacts['ComplexArtifact'].array == complex_artifact.array)
    assert loaded_artifact_store.artifacts['ComplexArtifact'].simple == complex_artifact.simple
