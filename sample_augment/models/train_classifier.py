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
from torch import nn, Tensor  # All neural network modules
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch.utils.data import DataLoader, WeightedRandomSampler  # Gives easier dataset management
from torch.utils.data import Dataset
from torchvision.transforms import transforms, Normalize
from tqdm import tqdm  # For nice progress bar!

from sample_augment.core import step, Artifact
from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.synth_data import SynthAugmentedTrain
from sample_augment.data.train_test_split import ValSet, TrainSet, stratified_split, stratified_k_fold_split
from sample_augment.models.classifier import VisionTransformer, ResNet50, DenseNet201, \
    EfficientNetV2  # or CustomDensenet, etc.
# from sample_augment.models.evaluate_classifier import ValidationPredictions
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
                balance_classes: bool, lr_schedule: bool,
                threshold_lambda: float,
                lr_step_size: int, lr_gamma: float) -> ClassifierMetrics:
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

    if lr_schedule:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size,
                                                    gamma=lr_gamma)
        log.info(f"Doing StepLR scheduling with step_size={lr_step_size} and gamma={lr_gamma}.")

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
        val_logits = []
        val_labels = []
        train_accuracies = 0

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

        if lr_schedule:
            # noinspection PyUnboundLocalVariable
            scheduler.step()
            log.info(f"Current learning rate: {optimizer.param_groups[0]['lr']:.6f}")
        # Validation Set evaluation at the end of this epoch:

        model.eval()

        for batch_idx, (image, label) in enumerate(val_loader):
            # move data to GPU if possible
            image = image.to(device=device)
            label = label.to(device=device)

            with torch.no_grad():
                logits = model(image)

            val_logits.append(logits)
            val_labels.append(label)
            # calculate validation metrics
            # val_losses += val_loss.item()
            # val_accuracies += ((predictions > threshold) == label).float().mean().item()
            # # if the model doesn't predict any class (zero_division), give it a score of 0
            # val_f1s += f1_score(label.cpu().numpy(),
            #                     (predictions > threshold).cpu().numpy(), average='macro',
            #                     zero_division=0)
        val_logits = torch.cat(val_logits, dim=0).cpu()
        val_labels = torch.cat(val_labels, dim=0).cpu()
        val_predictions = torch.sigmoid(val_logits)
        val_loss = criterion(val_logits, val_labels)
        assert isinstance(val_set, ValSet), "val_set should be ValSet for threshold"
        time_pre = time.time()
        # print(f"Predictions device: {val_predictions.device}")
        # print(f"ValSet label_tensor device: {val_set.label_tensor.device}")
        thresholds = determine_threshold_vector(val_predictions, val_set, threshold_lambda, n_support=100)
        time_post = time.time()
        log.debug(f"Threshold time: {time_post - time_pre:.2f}")

        binary_predictions = (val_predictions > thresholds).cpu()
        val_accuracy = (binary_predictions == val_labels.cpu()).float().mean().item()

        # Calculate F1 score, using the same thresholds
        val_f1 = f1_score(val_labels.cpu().numpy(),
                          binary_predictions.numpy(),
                          average='macro',
                          zero_division=0)

        val_loss_per_epoch.append(val_loss)
        val_acc_per_epoch.append(val_accuracy)
        val_f1_per_epoch.append(val_f1)

        # model checkpointing, take the best model so far
        if val_f1 > best_f1_score:
            best_epoch = epoch
            best_val_loss = val_loss
            best_f1_score = val_f1
            best_model_state = model.state_dict()  # Store the state dict of the best model so far

        log.info(
            f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {avg_train_loss:.2f}, '
            f'Val Loss: {val_loss:.2f}, '
            f'Val Acc: {val_accuracy:.2f}, '
            f'Val F1: {val_f1:.3f}')

    log.info(f"Classifier training: Best model from epoch {best_epoch + 1} with val_loss = {best_val_loss:.3f} "
             f"and f1_score = {best_f1_score:.3f}")

    # load model state from best performing epoch
    model.load_state_dict(best_model_state)
    return ClassifierMetrics(
        train_loss=np.array(train_loss_per_epoch),
        validation_loss=np.array(val_loss_per_epoch),
        train_accuracy=np.array(train_acc_per_epoch),
        validation_accuracy=np.array(val_acc_per_epoch),
        validation_f1=np.array(val_f1_per_epoch),
        epoch=best_epoch + 1
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
                     lr_schedule: bool,
                     threshold_lambda: float,
                     lr_step_size: int = 10, lr_gamma: float = 0.7) -> TrainedClassifier:
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
        random_crop = transforms.RandomResizedCrop(256 if not isinstance(model, VisionTransformer) else 224,
                                                   scale=(0.85, 1.0), **optional_aa_arg)

        geometric_transforms = [
            # antialias argument needed for newer versions of torchvision
            WithProbability(transform=CircularTranslate(shift_range=40), p=0.5),
            # WithProbability(transform=transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), p=0.5),
            WithProbability(transform=random_crop, p=0.5)
        ]
        if geometric_augment:
            # add geometric transforms to base_transforms (before ColorJitter)
            base_transforms[2:2] = geometric_transforms

        train_data.transform = transforms.Compose(base_transforms)
        val_data.transform = transforms.Compose(plain_transforms)
    else:
        train_data.transform = transforms.Compose(plain_transforms)
        val_data.transform = transforms.Compose(plain_transforms)

    if isinstance(model, VisionTransformer):
        log.debug(f"Using a VisionTransformer, resizing inputs to 224x224. (device={device})")
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
                          balance_classes=balance_classes, random_seed=random_seed,
                          lr_schedule=lr_schedule, threshold_lambda=threshold_lambda,
                          lr_step_size=lr_step_size, lr_gamma=lr_gamma)

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
                            lr_schedule: bool,
                            threshold_lambda: float,
                            lr_step_size: int = 10, lr_gamma: float = 0.7
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
        classifier: TrainedClassifier = train_classifier(TrainSet.from_existing(train, name=f"train_fold_{i}"),
                                                         ValSet.from_existing(val, name=f"val_fold_{i}"), model_type,
                                                         num_epochs, batch_size, learning_rate, balance_classes,
                                                         fold_random_seed, data_augment, geometric_augment,
                                                         color_jitter, h_flip_p, v_flip_p, lr_schedule=lr_schedule,
                                                         threshold_lambda=threshold_lambda, lr_step_size=lr_step_size,
                                                         lr_gamma=lr_gamma)
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
def train_augmented_classifier(train_data: SynthAugmentedTrain, val_data: ValSet,
                               num_epochs: int, batch_size: int, learning_rate: float,
                               balance_classes: bool,
                               model_type: ModelType,
                               random_seed: int,
                               data_augment: bool,
                               geometric_augment: bool,
                               color_jitter: float,
                               h_flip_p: float,
                               v_flip_p: float,
                               lr_schedule: bool,
                               threshold_lambda: float) -> SynthTrainedClassifier:
    # synth_training_set, val_set, num_epochs, batch_size, learning_rate, balance_classes,
    # random_seed, data_augment, geometric_augment, color_jitter, h_flip_p, v_flip_p, synth_p=synth_p
    return SynthTrainedClassifier(
        train_classifier(train_data, val_data, model_type, num_epochs, batch_size, learning_rate, balance_classes,
                         random_seed, data_augment, geometric_augment, color_jitter, h_flip_p, v_flip_p, lr_schedule,
                         threshold_lambda=threshold_lambda))


def determine_threshold_vector(predictions: Tensor, val: ValSet, threshold_lambda: float,
                               n_support: int = 250) -> Tensor:
    """
    Finds the optimal threshold for each class with respect to the F1 score on the validation set.
    Predictions should be in the range 0..1, so have sigmoid already applied.
    Returns:
      np.ndarray containing the optimal threshold for each class.
    """

    predictions = predictions.cpu().numpy()
    labels = val.label_tensor.cpu().numpy()

    num_classes = labels.shape[1]
    thresholds_for_class = np.linspace(0, 1, n_support)

    best_thresholds = []

    for class_idx in range(num_classes):
        # Using broadcasting, we create an array where each row represents the comparison with a different threshold
        binary_predictions = predictions[:, class_idx].reshape(-1, 1) > thresholds_for_class

        f1_scores = np.array([f1_score(labels[:, class_idx], binary_predictions[:, i])
                              for i in range(n_support)])

        # Apply regularization penalty
        f1_scores = f1_scores - threshold_lambda * np.abs(thresholds_for_class - 0.5)

        # Find the threshold that maximizes the score
        best_threshold = thresholds_for_class[np.argmax(f1_scores)]
        best_thresholds.append(best_threshold)

    return torch.FloatTensor(best_thresholds)
