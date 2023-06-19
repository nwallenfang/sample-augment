import sys
from typing import Dict

import numpy as np
import torch
import torchvision.models
from torch import nn  # All neural network modules
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch.utils.data import DataLoader  # Gives easier dataset management
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm  # For nice progress bar!

from sample_augment.core import step, Artifact
from sample_augment.data.train_test_split import ValSet, TrainSet
from sample_augment.utils import log


class CustomDenseNet(torchvision.models.DenseNet):
    num_classes: int

    def __init__(self, num_classes, load_pretrained=False):
        # Initialize with densenet201 configuration
        super().__init__(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                         num_classes=1000)  # initially set to match the pre-trained model
        if load_pretrained:
            # use the model pretrained on imagenet
            pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', weights='IMAGENET1K_V1')
            self.load_state_dict(pretrained.state_dict(), strict=False)
        # Freeze early layers
        for param in self.parameters():
            param.requires_grad = False

        # Modify the classifier part of the model
        self.classifier = nn.Sequential(
            nn.Linear(1920, 960),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(960, 240),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(240, 30),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(30, num_classes))

        self.num_classes = num_classes

    def get_kwargs(self) -> Dict:
        return {'num_classes': self.num_classes}


class ClassifierMetrics(Artifact):
    """
        collection of metrics taken over each training epoch.
    """
    train_loss: np.ndarray
    validation_loss: np.ndarray
    train_accuracy: np.ndarray
    validation_accuracy: np.ndarray


def preprocess(dataset):
    data = dataset.tensors[0]

    if data.dtype == torch.uint8:
        # convert to float32
        data = data.float()
        data /= 255.0

    # might need to preprocess to required resolution
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        # ImageNet normalization factors
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # convert labels to expected Long dtype as well
    dataset.tensors = (transform(data), dataset.tensors[1].long())

    return dataset


def train(train_set: Dataset, val_set: Dataset, model: nn.Module, num_epochs, batch_size,
          learning_rate) -> ClassifierMetrics:
    """
        this code is taken in large part from Michel's notebook,
        see references/Michel_99_base_line_DenseNet_201_PyTorch.ipynb
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # TODO set random_seed so the experiment is (more) reproducible
    # Train Network
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_per_epoch = []
    train_loss_per_batch = []

    val_loss_per_epoch = []
    val_loss_per_batch = []

    train_acc_per_epoch = []
    val_acc_per_epoch = []

    for epoch in range(num_epochs):
        model.train()
        train_loss = -1
        val_loss = -1
        val_accuracy = -1

        for batch_idx, (image, label) in enumerate(tqdm(train_loader, file=sys.stdout, desc="Training")):
            # Get data to cuda if possible
            image = image.to(device=device)
            label = label.to(device=device)

            # forward
            predictions = model(image)  # Pass batch
            train_loss = criterion(predictions, label)  # Calculate the loss

            # backward
            optimizer.zero_grad()  #
            train_loss.backward()  # Calculate the gradients

            # gradient descent or adam step
            optimizer.step()  # Update the weights

            # store loss
            train_loss_per_batch.append(train_loss.item())

        train_loss_per_epoch.append(sum(train_loss_per_batch) / len(train_loss_per_batch))
        train_loss_per_batch.clear()
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], '
            f' Train Loss: {train_loss.item():.4f}')

        model.eval()
        for batch_idx, (image, label) in enumerate(tqdm(test_loader, file=sys.stdout, desc="Validation")):
            # Move data to GPU if possible
            image = image.to(device=device)
            label = label.to(device=device)

            # forward
            with torch.no_grad():
                predictions = model(image)

            val_loss = criterion(predictions, label)
            val_accuracy = (predictions.argmax(dim=-1) == label).float().mean()

            # store metrics
            val_loss_per_batch.append(val_loss.item())

        # save checkpoints? not sure..
        # torch.save(model.state_dict(),
        #            project_path(f'models/checkpoints/densenet201/tmp-{epoch}-val-{val_loss:.3f}.pt'))

        val_loss_per_epoch.append(sum(val_loss_per_batch) / len(val_loss_per_batch))
        val_loss_per_batch.clear()
        log.info(
            f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Val Loss: {val_loss.item():.3f}, '
            f'Val Accuracy: {val_accuracy.item():.2f}')

    return ClassifierMetrics(
        train_loss=np.array(train_loss_per_epoch),
        validation_loss=np.array(val_loss_per_epoch),
        train_accuracy=np.array(train_acc_per_epoch),
        validation_accuracy=np.array(val_acc_per_epoch)
    )


class TrainedClassifier(Artifact):
    name: str
    model: torch.nn.Module
    # these following metrics are taken for each epoch
    # TODO extra cred task: a decorator like @extractable would be nice. This decorator would mean that a
    #  "virtual" step would be created that takes a TrainedClassifier and returns the subartifact
    #  ClassifierMetrics. Then in evaluate we can take just ClassiferMetrics as arg.
    metrics: ClassifierMetrics


@step
def train_classifier(train_data: TrainSet, val_data: ValSet,
                     num_epochs: int, batch_size: int, learning_rate: float) -> TrainedClassifier:
    """
    test the classifier training by training a Densenet201 on GC-10
    this code is taken in large part from Michel's notebook,
    see references/Michel_99_base_line_DenseNet_201_PyTorch.ipynb

    old default train params: num_epochs = 20, batch_size = 64, learning_rate = 0.001
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomDenseNet(num_classes=train_data.num_classes, load_pretrained=True)
    model.to(device)
    # train_data = AugmentDataset.load_from_file(Path(project_path('data/interim/gc10_train.pt')))
    # val_data = AugmentDataset.load_from_file(Path(project_path('data/interim/gc10_val.pt')))

    train_data = preprocess(train_data)
    val_data = preprocess(val_data)

    metrics = train(train_data, val_data, model, num_epochs=num_epochs,
                    batch_size=batch_size, learning_rate=learning_rate)

    return TrainedClassifier(
        name="missing name",
        model=model,
        metrics=metrics
    )
