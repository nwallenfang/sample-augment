import inspect
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
from sample_augment.data.synth_augment import TrainSetWithSynthetic
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


class CustomResNet50(torchvision.models.ResNet):
    def __init__(self, num_classes, load_pretrained=False):
        # initially set num_classes to 1000 to match the pre-trained model
        super(CustomResNet50, self).__init__(block=torchvision.models.resnet.Bottleneck,
                                             layers=[3, 4, 6, 3], num_classes=1000)

        # Freeze early layers
        if load_pretrained:
            # use the model pretrained on imagenet
            pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', weights='IMAGENET1K_V2')
            self.load_state_dict(pretrained.state_dict(), strict=False)
        for param in self.parameters():
            param.requires_grad = False

        # Modify the classifier part of the model
        self.fc = nn.Sequential(
            nn.Linear(2048, 960),
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

    def get_kwargs(self):
        return {'num_classes': self.num_classes}


class CustomViT(torchvision.models.VisionTransformer):
    def __init__(self, num_classes, load_pretrained=False):
        # use same settings as vit_b_16 to get the same architecture
        super().__init__(
            image_size=224,
            patch_size=16,
            num_layers=12,
            num_heads=12,
            hidden_dim=768,
            mlp_dim=3072,
            num_classes=1000,
            representation_size=None)

        # Load the pretrained weights if requested
        if load_pretrained:
            # use the model pretrained on imagenet
            weights = torchvision.models.vision_transformer.ViT_B_16_Weights.IMAGENET1K_V1
            # pretrained = torch.hub.load('pytorch/vision:v0.10.0', 'vit_b_16', weights='IMAGENET1K_V1')
            self.load_state_dict(weights.get_state_dict(progress=True), strict=False)

        # Freeze the early layers
        for param in self.parameters():
            param.requires_grad = False

        # Create a custom classifier head
        self.heads = nn.Sequential(
            nn.Linear(768, 960),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(960, 240),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(240, 30),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(30, num_classes)
        )
        self.num_classes = num_classes

        # Unfreeze the custom classifier head
        for param in self.heads.parameters():
            param.requires_grad = True

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


# Quick sanity check
# for i in range(10):
#     img = inverse_normalize(image[i].cpu()).numpy()  # Take the first image of the batch
#     plt.imshow(img.transpose(1, 2, 0))  # Assuming the image has shape (channels, height, width)
#     plt.title(f'Label: {label[i]}')
#     plt.axis('off')
#     plt.show()

def train_model(train_set: Dataset, val_set: Dataset, model: nn.Module, num_epochs: int, batch_size: int,
                learning_rate: float, random_seed: int,
                balance_classes: bool, threshold: float) -> ClassifierMetrics:
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
            # train_accuracies += (predictions.argmax(dim=-1) == label).float().mean().item()
            train_accuracies += ((predictions > threshold) == label).float().mean().item()

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
                logits = model(image)
                predictions = torch.sigmoid(logits)

            val_loss = criterion(logits, label)

            # calculate validation metrics
            val_losses += val_loss.item()
            val_accuracies += ((predictions > threshold) == label).float().mean().item()
            # if the model doesn't predict any class (zero_division), give it a score
            # of 0
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

    log.info(f"Classifier training: Best model from epoch {best_epoch} with val_loss = {best_val_loss:.3f} "
             f"and f1_score = {best_f1_score:.3f}")

    # load model state from best performing epoch
    model.load_state_dict(best_model_state)
    return ClassifierMetrics(
        train_loss=np.array(train_loss_per_epoch),
        validation_loss=np.array(val_loss_per_epoch),
        train_accuracy=np.array(train_acc_per_epoch),
        validation_accuracy=np.array(val_acc_per_epoch),
        validation_f1=np.array(val_f1_per_epoch),
        epoch=best_epoch
    )


class TrainedClassifier(Artifact):
    # name: str
    model: torch.nn.Module
    # these following metrics are taken for each epoch
    # extra cred task: a decorator like @extractable would be nice. This decorator would mean that a
    #  "virtual" step would be created that takes a TrainedClassifier and returns the subartifact
    #  ClassifierMetrics. Then in evaluate we can take just ClassiferMetrics as arg.
    metrics: ClassifierMetrics


plain_transforms = [
    transforms.ConvertImageDtype(dtype=torch.float32),
    normalize
]


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


# class AugmentedDataset(AugmentDataset):
#     def __init__(self, original_with_synthetic: TrainSetWithSynthetic, synth_p: float, name: str,
#                  tensors: Tuple[Tensor, Tensor], img_ids: List[str], root_dir: Path, **kwargs):
#         super().__init__(name, tensors, img_ids, root_dir, **kwargs)
#         self.original_with_synthetic = original_with_synthetic
#         self.synth_p = synth_p
#
#     def __len__(self):
#         return len(self.original_with_synthetic)
#
#     def __getitem__(self, idx):
#         image, label = self.original_with_synthetic.__getitem__(idx)
#         if np.random.rand() < self.synth_p:
#             same_class_indices = (self.original_with_synthetic.synthetic_labels == label).nonzero(as_tuple=True)[0]
#             if len(same_class_indices) > 0:  # If there are synthetic instances from the same class
#                 # Replace the real instance with a synthetic one
#                 chosen_idx = np.random.choice(same_class_indices)
#                 image, label = (self.original_with_synthetic.synthetic_images[chosen_idx],
#                                 self.original_with_synthetic.synthetic_labels[chosen_idx])
#                 # Apply the same transformation to the synthetic image
#                 if self.original_with_synthetic.transform is not None:
#                     image = self.original_with_synthetic.transform(image)
#                 else:
#                     log.warning("synthetic dataset transform is None!")
#         return image, label


@step
def train_classifier(train_data: TrainSet, val_data: ValSet,
                     num_epochs: int, batch_size: int, learning_rate: float,
                     balance_classes: bool,
                     random_seed: int,
                     data_augment: bool,
                     geometric_augment: bool,
                     color_jitter: float,
                     h_flip_p: float,
                     v_flip_p: float,
                     threshold: float,
                     synth_p: float) -> TrainedClassifier:
    """
    test the classifier training by training a Densenet201 on GC-10
    this code is taken in large part from Michel's notebook,
    see references/Michel_99_base_line_DenseNet_201_PyTorch.ipynb

    old default train params: num_epochs = 20, batch_size = 64, learning_rate = 0.001
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = CustomDenseNet(num_classes=train_data.num_classes, load_pretrained=True)
    # model = CustomResNet50(num_classes=train_data.num_classes, load_pretrained=True)
    model = CustomViT(num_classes=train_data.num_classes, load_pretrained=True)
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
        # else:
        #     random_crop = transforms.RandomResizedCrop(256, scale=(0.85, 1.0))

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

    if isinstance(model, CustomViT):
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
                          balance_classes=balance_classes, random_seed=random_seed, threshold=threshold)

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
                            v_flip_p: float
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

        classifier: TrainedClassifier = train_classifier(TrainSet.from_existing(train, name="train_fold_{i}"),
                                                         ValSet.from_existing(val, name="val_fold_{i}"),
                                                         num_epochs, batch_size, learning_rate,
                                                         balance_classes, fold_random_seed, data_augment,
                                                         geometric_augment, color_jitter, h_flip_p, v_flip_p)
        classifiers.append(classifier)

    return KFoldTrainedClassifiers(classifiers=classifiers)


class SynthTrainedClassifier(TrainedClassifier):
    # noinspection PyMissingConstructor
    def __init__(self, trained_classifier: TrainedClassifier, **kwargs):
        # Copy all attributes from the superclass instance to the subclass instance
        super().__init__(model=trained_classifier.model, metrics=trained_classifier.metrics, **kwargs)


@step
def train_augmented_classifier(train_data: TrainSetWithSynthetic, val_data: ValSet,
                               num_epochs: int, batch_size: int, learning_rate: float,
                               balance_classes: bool,
                               random_seed: int,
                               data_augment: bool,
                               geometric_augment: bool,
                               color_jitter: float,
                               h_flip_p: float,
                               v_flip_p: float,
                               threshold: float,
                               synth_p: float) -> SynthTrainedClassifier:
    return SynthTrainedClassifier(
        train_classifier(train_data, val_data, num_epochs, batch_size, learning_rate, balance_classes, random_seed,
                         data_augment, geometric_augment, color_jitter, h_flip_p, v_flip_p, threshold, synth_p))
