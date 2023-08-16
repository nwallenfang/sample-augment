import inspect
import random
import sys
import time
from copy import deepcopy
from enum import Enum
from typing import List

import numpy as np
import torch
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
from sample_augment.models.classifier import VisionTransformer, ResNet50, DenseNet201, \
    EfficientNetV2  # or CustomDensenet, etc.
from sample_augment.sampling.synth_augment import SynthAugTrainSet
from sample_augment.utils import log

_mean = torch.tensor([0.485, 0.456, 0.406])
_std = torch.tensor([0.229, 0.224, 0.225])
normalize = Normalize(mean=_mean, std=_std)
# reverse operation for use in visualization
inverse_normalize = Normalize((-_mean / _std).tolist(), (1.0 / _std).tolist())


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
    epoch: int


def create_weighted_random_sampler(dataset: Dataset):
    # create a weighted sampler that oversamples smaller classes to make each class
    # appear at the same frequency

    # calling len on dataset is fine
    # noinspection PyTypeChecker
    assert isinstance(dataset, AugmentDataset)

    labels = [dataset.primary_label_tensor[i] for i in range(len(dataset))]
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


def train_model(train_set: Dataset, val_set: Dataset, model: nn.Module, num_epochs: int, batch_size: int,
                learning_rate: float, random_seed: int,
                balance_classes: bool) -> ClassifierMetrics:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # make the experiment as reproducible as possible by setting a random seed
    _set_random_seed(random_seed)

    sampler = create_weighted_random_sampler(train_set) if balance_classes else None
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=(sampler is None),
                              sampler=sampler)
    val_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
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

    # fixed threshold just during training, later we'll calculate the optimal one
    threshold = 0.5

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
            logits = model(image)  # Pass batch
            # get sigmoid probabilities
            predictions = torch.sigmoid(logits)

            train_loss = criterion(logits, label)  # Calculate the loss
            # fixed threshold just during training, later we'll calculate the optimal one
            train_accuracies += ((predictions > threshold) == label).float().mean().item()

            # backward
            optimizer.zero_grad()
            train_loss.backward()  # Calculate the gradients

            # gradient descent or adam step
            optimizer.step()  # Update the weights

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
                logits = model(image)
                predictions = torch.sigmoid(logits)

            val_loss = criterion(logits, label)

            # calculate validation metrics
            val_losses += val_loss.item()
            val_accuracies += ((predictions > threshold) == label).float().mean().item()
            # if the model doesn't predict any class (zero_division), give it a score of 0
            val_f1s += f1_score(label.cpu().numpy(),
                                (predictions > threshold).cpu().numpy(), average='macro',
                                zero_division=0)

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

    log.info(f"Classifier training: Best model from epoch {best_epoch+1} with val_loss = {best_val_loss:.3f} "
             f"and f1_score = {best_f1_score:.3f}")

    # load model state from best performing epoch
    model.load_state_dict(best_model_state)
    return ClassifierMetrics(
        train_loss=np.array(train_loss_per_epoch),
        validation_loss=np.array(val_loss_per_epoch),
        train_accuracy=np.array(train_acc_per_epoch),
        validation_accuracy=np.array(val_acc_per_epoch),
        validation_f1=np.array(val_f1_per_epoch),
        epoch=best_epoch+1
    )


class TrainedClassifier(Artifact):
    model: torch.nn.Module
    metrics: ClassifierMetrics


class WithProbability:
    """
        call a transform with a certain probability, else do nothing
    """

    def __init__(self, transform, p=0.5):
        self.transform = transform
        self.probability = p

    def __call__(self, x):
        if torch.rand(1).item() < self.probability:
            x = self.transform(x)
        return x


class CircularTranslate:
    def __init__(self, shift_range):
        self.shift_range = shift_range

    def __call__(self, img):
        shift_x = random.randint(-self.shift_range, self.shift_range)
        shift_y = random.randint(-self.shift_range, self.shift_range)
        img = torch.roll(img, shifts=(shift_y, shift_x), dims=(1, 2))  # Shift along height and width dimensions
        return img


plain_transforms = [
    transforms.ConvertImageDtype(dtype=torch.float32),
    normalize
]


class ModelType(str, Enum):
    DenseNet = "DenseNet"
    ResNet = "ResNet"
    VisionTransformer = "VisionTransformer"
    EfficientNetV2_S = "EfficientNetV2_S"
    EfficientNetV2_L = "EfficientNetV2_L"


@step
def train_classifier(train_data: TrainSet, val_data: ValSet, model_type: ModelType,
                     num_epochs: int, batch_size: int, learning_rate: float,
                     balance_classes: bool,
                     random_seed: int,
                     data_augment: bool,
                     geometric_augment: bool,
                     color_jitter: float,
                     h_flip_p: float,
                     v_flip_p: float,
                     synth_p: float) -> TrainedClassifier:
    """
    test the classifier training by training a Densenet201 on GC-10
    this code is taken in large part from Michel's notebook,
    see references/Michel_99_base_line_DenseNet_201_PyTorch.ipynb

    old default train params: num_epochs = 20, batch_size = 64, learning_rate = 0.001
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if model_type == ModelType.ResNet:
        model = ResNet50(num_classes=train_data.num_classes, load_pretrained=True)
    elif model_type == ModelType.DenseNet:
        model = DenseNet201(num_classes=train_data.num_classes, load_pretrained=True)
    elif model_type == ModelType.VisionTransformer:
        model = VisionTransformer(num_classes=train_data.num_classes, load_pretrained=True)
    elif model_type == ModelType.EfficientNetV2_S:
        model = EfficientNetV2(num_classes=train_data.num_classes, size='S', load_pretrained=True)
    elif model_type == ModelType.EfficientNetV2_L:
        model = EfficientNetV2(num_classes=train_data.num_classes, size='L', load_pretrained=True)
    else:
        # noinspection PyUnresolvedReferences
        raise ValueError(f'Invalid model_type `{model_type}`. Available models: {", ".join(e.name for e in ModelType)}')
    log.debug(f"Training {model_type}")
    model.to(device)
    train_data = deepcopy(train_data)
    val_data = deepcopy(val_data)
    antialias_param_needed = 'antialias' in inspect.getfullargspec(transforms.RandomResizedCrop).args
    optional_aa_arg = {"antialias": True} if antialias_param_needed else {}
    if data_augment:
        base_transforms = [
            transforms.ConvertImageDtype(dtype=torch.float32),
            transforms.RandomVerticalFlip(p=v_flip_p),
            transforms.RandomHorizontalFlip(p=h_flip_p),
            transforms.ColorJitter(brightness=color_jitter, contrast=color_jitter, saturation=color_jitter),
            normalize,
        ]

        # if antialias_param_needed:  # needed in newer torchvision versions
        # noinspection PyArgumentList
        random_crop = transforms.RandomResizedCrop(256, scale=(0.85, 1.0), **optional_aa_arg)

        geometric_transforms = [
            # antialias argument needed for newer versions of torchvision
            WithProbability(transform=CircularTranslate(shift_range=40), p=0.5),
            # WithProbability(transform=transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), p=0.5),
            WithProbability(transform=random_crop, p=0.5)
        ]
        if geometric_augment:
            # add geometric transforms to base_transforms (before ColorJitter)
            base_transforms[2:2] = geometric_transforms

        # add synthetic augmentation if synthetic data is present
        # can't use isinstance due to the pesky importing / class use issue....
        if train_data.__full_name__ == 'data.synth_augment.TrainSetWithSynthetic':
            log.info(f"Doing synthetic Data Augmentation with probability {synth_p}.")

        train_data.transform = transforms.Compose(base_transforms)
        val_data.transform = transforms.Compose(plain_transforms)
    else:
        train_data.transform = transforms.Compose(plain_transforms)
        val_data.transform = transforms.Compose(plain_transforms)

    if isinstance(model, VisionTransformer):
        log.info("Using a VisionTransformer, resizing inputs to 224x224.")
        train_data.transform = transforms.Compose([
            transforms.Resize((224, 224), **optional_aa_arg),
            train_data.transform
        ])
        val_data.transform = transforms.Compose([
            transforms.Resize((224, 224), **optional_aa_arg),
            val_data.transform
        ])

    metrics = train_model(train_data, val_data, model, num_epochs=num_epochs,
                          batch_size=batch_size, learning_rate=learning_rate,
                          balance_classes=balance_classes, random_seed=random_seed)

    del train_data
    del val_data
    return TrainedClassifier(
        model=model,
        metrics=metrics
    )


class KFoldTrainedClassifiers(Artifact):
    classifiers: List[TrainedClassifier]


@step
def k_fold_train_classifier(dataset: AugmentDataset, n_folds: int,
                            test_ratio: float,
                            model_type: ModelType,
                            min_instances: int,
                            random_seed: int,
                            num_epochs: int,
                            batch_size: int,
                            learning_rate: float,
                            balance_classes: bool,
                            data_augment: bool,
                            geometric_augment: bool,
                            color_jitter: float,
                            h_flip_p: float,
                            v_flip_p: float,
                            synth_p: float
                            ) -> KFoldTrainedClassifiers:
    fold_random_seed = random_seed
    train_val, test = stratified_split(dataset, 1.0 - test_ratio, random_seed, min_instances)
    classifiers = []
    splits = stratified_k_fold_split(train_val, n_folds, random_seed, min_instances)
    for i, (train, val) in enumerate(splits.datasets):
        # different random seed when splitting for each fold
        # noinspection PyTypeChecker
        log.info(f"Training classifier on fold {i + 1}")
        fold_random_seed += 1

        start_time = time.time()  # Start the timer
        classifier: TrainedClassifier = train_classifier(TrainSet.from_existing(train, name="train_fold_{i}"),
                                                         ValSet.from_existing(val, name="val_fold_{i}"),
                                                         model_type,
                                                         num_epochs, batch_size, learning_rate,
                                                         balance_classes, fold_random_seed, data_augment,
                                                         geometric_augment, color_jitter, h_flip_p, v_flip_p,
                                                         synth_p)
        classifiers.append(classifier)
        elapsed_time = time.time() - start_time  # Calculate elapsed time
        minutes, seconds = divmod(elapsed_time, 60)  # Split elapsed time into minutes and seconds
        # noinspection PyTypeChecker
        log.info(f"Fold {i + 1} completed in {int(minutes)}m{int(seconds)}s")
    return KFoldTrainedClassifiers(classifiers=classifiers)


class SynthTrainedClassifier(TrainedClassifier):
    # noinspection PyMissingConstructor
    def __init__(self, trained_classifier: TrainedClassifier, **kwargs):
        # Copy all attributes from the superclass instance to the subclass instance
        super().__init__(model=trained_classifier.model, metrics=trained_classifier.metrics, **kwargs)


@step
def train_augmented_classifier(train_data: SynthAugTrainSet, val_data: ValSet,
                               num_epochs: int, batch_size: int, learning_rate: float,
                               balance_classes: bool,
                               model_type: ModelType,
                               random_seed: int,
                               data_augment: bool,
                               geometric_augment: bool,
                               color_jitter: float,
                               h_flip_p: float,
                               v_flip_p: float,
                               synth_p: float) -> SynthTrainedClassifier:
    # synth_training_set, val_set, num_epochs, batch_size, learning_rate, balance_classes,
    # random_seed, data_augment, geometric_augment, color_jitter, h_flip_p, v_flip_p, synth_p=synth_p
    return SynthTrainedClassifier(
        train_classifier(train_data, val_data, model_type, num_epochs, batch_size, learning_rate, balance_classes,
                         random_seed, data_augment, geometric_augment, color_jitter, h_flip_p, v_flip_p, synth_p))
