import torch

from sample_augment.core import Artifact


class TrainedClassifier(Artifact):
    name: str
    model: torch.nn.Module
