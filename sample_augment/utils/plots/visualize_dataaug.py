import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms

from sample_augment.core import step
from sample_augment.data.dataset import AugmentDataset
from sample_augment.models.train_classifier import WithProbability, CircularTranslate


@step
def visualize_data_augmentation(train_data: AugmentDataset):
    base_transforms = [
        transforms.ConvertImageDtype(dtype=torch.float32),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.55, contrast=0.55, saturation=0.55),
        # normalize,
    ]
    geometric_transforms = [
        # antialias argument needed for newer versions of torchvision
        WithProbability(transform=CircularTranslate(shift_range=40), p=0.5),
        # WithProbability(transform=transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)), p=0.5),
        WithProbability(transform=transforms.RandomResizedCrop(256, scale=(0.85, 1.0)), p=0.5),
    ]
    # add geometric transforms to base_transforms (before ColorJitter)
    base_transforms[2:2] = geometric_transforms

    augment_transforms = transforms.Compose(base_transforms)

    # train_data.transform = augment_transforms

    # Assume that `dataset` is your dataset and `augment_transforms` is your augmentation pipeline
    # Let's say we want to sample the first instance in the dataset
    image, label = train_data[0]

    augmented_images = []
    for _ in range(8):  # generate 8 augmented versions of the image
        augmented_image = augment_transforms(image)
        augmented_images.append(augmented_image)

    # Make a grid from the images
    grid = torchvision.utils.make_grid(augmented_images, nrow=4)  # 4 images per row

    # Convert the tensor to a numpy array and correct the color channel order
    np_grid = grid.permute(1, 2, 0).numpy()

    # Display the grid
    plt.figure(figsize=(10, 10))
    plt.imshow(np_grid)
    plt.show()
