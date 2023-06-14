import torch
from torch import nn  # All neural network modules
from torch import optim  # For optimizers like SGD, Adam, etc.
from torch.utils.data import DataLoader  # Gives easier dataset management
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm  # For nice progress bar!

from sample_augment.core import step
from sample_augment.data.train_test_split import ValSet, TrainSet
from sample_augment.models.classifier import TrainedClassifier
from sample_augment.utils.paths import project_path


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


def train(train_set: Dataset, val_set: Dataset, model: nn.Module, num_epochs=23, batch_size=64,
          learning_rate=0.001):
    """
        this code is taken in large part from Michel's notebook,
        see references/Michel_99_base_line_DenseNet_201_PyTorch.ipynb
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # this DataParallel approach doesn't work on windows. If we want to accelerate with 2 GPUS,
    # we need to use DDP:
    # https://cloudblogs.microsoft.com/opensource/2021/08/04/introducing-distributed-data-parallel-support-on-pytorch-windows/
    # if torch.cuda.device_count() > 1:
    #   print("Let's use", torch.cuda.device_count(), "GPUs!")
    #   model = nn.DataParallel(model)

    # TODO set random_seed so the experiment is (more) reproducible
    # Train Network
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=val_set, batch_size=batch_size, shuffle=True)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_losses = []
    train_batch_losses = []

    val_losses = []
    val_batch_losses = []

    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        model.train()
        batch_idx = 0
        train_loss = -1
        val_loss = -1
        val_accuracy = -1

        for batch_idx, (image, label) in enumerate(tqdm(train_loader)):
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
            train_batch_losses.append(train_loss.item())

        train_losses.append(sum(train_batch_losses) / len(train_batch_losses))
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{n_total_steps}],'
            f' Train Loss: {train_loss.item():.4f}')

        model.eval()
        for batch_idx, (image, label) in enumerate(tqdm(test_loader)):
            # Get data to cuda if possible
            image = image.to(device=device)
            label = label.to(device=device)

            # forward
            with torch.no_grad():
                predictions = model(image)  # Pass batch

            val_loss = criterion(predictions, label)  # Calculate the loss
            val_accuracy = (predictions.argmax(dim=-1) == label).float().mean()
            # store loss
            val_batch_losses.append(val_loss.item())

        torch.save(model.state_dict(),
                   project_path(f'models/checkpoints/densenet201/tmp-{epoch}-val-{val_loss:.3f}.pt'))

        val_losses.append(sum(val_batch_losses) / len(val_batch_losses))
        print(
            f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{n_total_steps}], '
            f'Val Loss: {val_loss.item():.3f}, '
            f'Val Accuracy: {val_accuracy.item():.2f}')


@step
def train_classifier(train_data: TrainSet, val_data: ValSet) -> TrainedClassifier:
    """
    test the classifier training by training a Densenet201 on GC-10
    this code is taken in large part from Michel's notebook,
    see references/Michel_99_base_line_DenseNet_201_PyTorch.ipynb
    """
    num_classes = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # use the model pretrained on imagenet
    model = torch.hub.load('pytorch/vision:v0.10.0', 'densenet201', weights='IMAGENET1K_V1')

    # Freeze early layers
    for param in model.parameters():
        param.requires_grad = False

    # Modify the classifier part of the model
    model.classifier = nn.Sequential(
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
    model.to(device)

    # train_data = AugmentDataset.load_from_file(Path(project_path('data/interim/gc10_train.pt')))
    # val_data = AugmentDataset.load_from_file(Path(project_path('data/interim/gc10_val.pt')))

    train_data = preprocess(train_data)
    val_data = preprocess(val_data)

    train(train_data, val_data, model)

    return TrainedClassifier(
        name="missing name",
        model=model
    )
