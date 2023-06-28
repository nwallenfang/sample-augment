import random
import sys
from copy import deepcopy
from typing import Dict, List

import numpy as np
import torch
import torchvision.models
from sklearn.metrics import f1_score
from torch import nn  # All neural network modules
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch.utils.data import DataLoader, WeightedRandomSampler  # Gives easier dataset management
from torch.utils.data import Dataset
from torchvision.transforms import transforms, Normalize
from tqdm import tqdm  # For nice progress bar!

from sample_augment.core import step, Artifact
from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.train_test_split import ValSet, TrainSet, stratified_split, stratified_k_fold_split
from sample_augment.utils import log

_mean = torch.tensor([0.485, 0.456, 0.406])
_std = torch.tensor([0.229, 0.224, 0.225])
normalize = Normalize(mean=_mean, std=_std)
# reverse operation for use in visualization
inverse_normalize = Normalize((-_mean / _std).tolist(), (1.0 / _std).tolist())


class CustomDenseNet(torchvision.models.DenseNet):
    num_classes: int

    def __init__(self, num_classes, load_pretrained=False):
        # Initialize with densenet201 configuration
        super().__init__(num_init_features=64, growth_rate=32, block_config=(6, 12, 48, 32),
                         num_classes=1000)  # initially set num_classes to 1000 to match the pre-trained model
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
    validation_f1: np.ndarray
    """the epoch from which the final model state was selected"""
    # TODO add this
    # epoch: int


# could be moved to a more central location
# def preprocess(dataset: AugmentDataset) -> AugmentDataset:
#     data = dataset.tensors[0]
#     assert data.dtype == torch.uint8, f"preprocess() expected imgs to be uint8, got {data.dtype}"
#
#     # convert to float32
#     data = data.float()
#     data /= 255.0
#
#     # ImageNet normalization factors
#     # convert labels to expected Long dtype as well
#     # we're modifying the dataset here.
#     dataset.tensors = (normalize(data), dataset.tensors[1].long())
#
#     return dataset


def create_weighted_random_sampler(dataset: Dataset):
    # create a weighted sampler that oversamples smaller classes to make each class
    # appear at the same frequency

    # calling len on dataset is fine
    # noinspection PyTypeChecker
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    num_samples = len(labels)

    # Compute weight for each class and for each sample
    class_weights = [num_samples / class_counts[i] for i in range(len(class_counts))]
    weights = [class_weights[labels[i]] for i in range(num_samples)]

    # Create a sampler with these weights
    sampler = WeightedRandomSampler(weights, num_samples)
    return sampler


def _set_random_seed(random_seed: int):
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


# Quick sanity check
# for i in range(10):
#     img = inverse_normalize(image[i].cpu()).numpy()  # Take the first image of the batch
#     plt.imshow(img.transpose(1, 2, 0))  # Assuming the image has shape (channels, height, width)
#     plt.title(f'Label: {label[i]}')
#     plt.axis('off')
#     plt.show()

def train_model(train_set: Dataset, val_set: Dataset, model: nn.Module, num_epochs: int, batch_size: int,
                learning_rate: float, random_seed: int,
                balance_classes: bool) -> ClassifierMetrics:
    """
        this code is taken in large part from Michel's notebook,
        see docs/Michel_99_base_line_DenseNet_201_PyTorch.ipynb
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # make the experiment as reproducible as possible by setting a random seed
    _set_random_seed(random_seed)

    sampler = create_weighted_random_sampler(train_set) if balance_classes else None
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=(sampler is None),
                              sampler=sampler)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loss_per_epoch = []
    val_loss_per_epoch = []
    train_acc_per_epoch = []
    val_acc_per_epoch = []

    best_val_loss = float("inf")
    best_f1_score = -float("inf")  # Track best F1 score instead of best loss
    best_model_state = None
    val_f1_per_epoch = []

    best_epoch = -1

    for epoch in range(num_epochs):
        model.train()
        train_losses = 0
        val_losses = 0
        train_accuracies = 0
        val_accuracies = 0
        val_f1s = 0

        for batch_idx, (image, label) in enumerate(tqdm(train_loader, file=sys.stdout, desc="Training")):
            # move data to GPU if possible
            image = image.to(device=device)
            label = label.to(device=device)

            # forward
            predictions = model(image)  # Pass batch
            train_loss = criterion(predictions, label)  # Calculate the loss
            train_accuracies += (predictions.argmax(dim=-1) == label).float().mean().item()

            # backward
            optimizer.zero_grad()
            train_loss.backward()  # Calculate the gradients

            # gradient descent or adam step
            optimizer.step()  # Update the weights

            # store loss
            train_losses += train_loss.item()

        avg_train_loss = train_losses / len(train_loader)
        train_loss_per_epoch.append(avg_train_loss)
        train_acc_per_epoch.append(train_accuracies / len(train_loader))

        model.eval()
        for batch_idx, (image, label) in enumerate(val_loader):
            # move data to GPU if possible
            image = image.to(device=device)
            label = label.to(device=device)

            with torch.no_grad():
                predictions = model(image)

            val_loss = criterion(predictions, label)

            # calculate validation metrics
            val_losses += val_loss.item()
            val_accuracies += (predictions.argmax(dim=-1) == label).float().mean().item()
            val_f1s += f1_score(label.cpu().numpy(),
                                predictions.argmax(dim=-1).cpu().numpy(), average='macro')

        avg_val_loss = val_losses / len(val_loader)
        val_loss_per_epoch.append(avg_val_loss)
        avg_accuracy = val_accuracies / len(val_loader)
        val_acc_per_epoch.append(avg_accuracy)
        avg_f1 = val_f1s / len(val_loader)
        val_f1_per_epoch.append(avg_f1)

        if avg_f1 > best_f1_score:
            best_epoch = epoch
            best_val_loss = avg_val_loss
            best_f1_score = avg_f1
            best_model_state = model.state_dict()  # Store the state dict of the best model so far

        log.info(
            f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {avg_train_loss:.2f}, '
            f'Val Loss: {avg_val_loss:.2f}, '
            f'Val Acc: {avg_accuracy:.2f}, '
            f'Val F1: {avg_f1:.3f}')

    log.info(f"Classifier training: Best model from epoch {best_epoch} with val_loss = {best_val_loss:.3f} "
             f"and f1_score = {best_f1_score:.3f}")

    # load model state from best performing epoch
    model.load_state_dict(best_model_state)
    return ClassifierMetrics(
        train_loss=np.array(train_loss_per_epoch),
        validation_loss=np.array(val_loss_per_epoch),
        train_accuracy=np.array(train_acc_per_epoch),
        validation_accuracy=np.array(val_acc_per_epoch),
        validation_f1=np.array(val_f1_per_epoch)
    )


class TrainedClassifier(Artifact):
    # name: str
    model: torch.nn.Module
    # these following metrics are taken for each epoch
    # extra cred task: a decorator like @extractable would be nice. This decorator would mean that a
    #  "virtual" step would be created that takes a TrainedClassifier and returns the subartifact
    #  ClassifierMetrics. Then in evaluate we can take just ClassiferMetrics as arg.
    metrics: ClassifierMetrics


plain_transforms = transforms.Compose([
    transforms.ConvertImageDtype(dtype=torch.float32),
    normalize
])


@step
def train_classifier(train_data: TrainSet, val_data: ValSet,
                     num_epochs: int, batch_size: int, learning_rate: float,
                     balance_classes: bool,
                     random_seed: int,
                     data_augment: bool,
                     color_jitter: float,
                     h_flip_p: float,
                     v_flip_p: float) -> TrainedClassifier:
    """
    test the classifier training by training a Densenet201 on GC-10
    this code is taken in large part from Michel's notebook,
    see references/Michel_99_base_line_DenseNet_201_PyTorch.ipynb

    old default train params: num_epochs = 20, batch_size = 64, learning_rate = 0.001
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomDenseNet(num_classes=train_data.num_classes, load_pretrained=True)
    model.to(device)
    train_data = deepcopy(train_data)
    val_data = deepcopy(val_data)

    if data_augment:
        augment_transforms = transforms.Compose([
            transforms.ConvertImageDtype(dtype=torch.float32),
            transforms.RandomVerticalFlip(p=v_flip_p),
            transforms.RandomHorizontalFlip(p=h_flip_p),
            # transforms.RandomRotation(180),  # would rotate randomly from within (-180, 180),
            # which is not what we want
            transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter,
                                   saturation=color_jitter),
            normalize
        ])
        train_data.transform = augment_transforms
        val_data.transform = plain_transforms
    else:
        train_data.transform = plain_transforms
        val_data.transform = plain_transforms

    # preprocess / normalize AFTER augmentation transforms
    # train_data = typing.cast(TrainSet, preprocess(deepcopy(train_data)))
    # val_data = typing.cast(ValSet, preprocess(deepcopy(val_data)))

    metrics = train_model(train_data, val_data, model, num_epochs=num_epochs,
                          batch_size=batch_size, learning_rate=learning_rate,
                          balance_classes=balance_classes, random_seed=random_seed)

    return TrainedClassifier(
        model=model,
        metrics=metrics
    )


class KFoldTrainedClassifiers(Artifact):
    classifiers: List[TrainedClassifier]


@step
def k_fold_train_classifier(dataset: AugmentDataset, n_folds: int,
                            test_ratio: float,
                            min_instances: int,
                            random_seed: int,
                            num_epochs: int,
                            batch_size: int,
                            learning_rate: float,
                            balance_classes: bool,
                            data_augment: bool,
                            color_jitter: float,
                            h_flip_p: float,
                            v_flip_p: float
                            ) -> KFoldTrainedClassifiers:
    fold_random_seed = random_seed
    train_val, test = stratified_split(dataset, 1.0 - test_ratio, random_seed, min_instances)
    classifiers = []
    splits = stratified_k_fold_split(train_val, n_folds, random_seed, min_instances)
    for i, (train, val) in enumerate(splits.datasets):
        # different random seed when splitting for each fold
        log.info(f"Training classifier on fold {i + 1}")
        fold_random_seed += 1

        classifier: TrainedClassifier = train_classifier(TrainSet.from_existing(train, name="train_fold_{i}"),
                                                         ValSet.from_existing(val, name="val_fold_{i}"),
                                                         num_epochs, batch_size, learning_rate,
                                                         balance_classes, fold_random_seed, data_augment,
                                                         color_jitter, h_flip_p, v_flip_p)
        classifiers.append(classifier)

    return KFoldTrainedClassifiers(classifiers=classifiers)
