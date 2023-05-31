from abc import abstractmethod, ABC
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
import torch.nn as nn
import torchmetrics
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Normalize, ToPILImage
from tqdm import tqdm

from data.dataset import CustomTensorDataset
from models.DenseNetClassifier import DenseNet201
from utils.paths import project_path

_mean = torch.tensor([0.485, 0.456, 0.406])
_std = torch.tensor([0.229, 0.224, 0.225])
normalize = Normalize(mean=_mean, std=_std)
# reverse operation for use in visualization
inverse_normalize = Normalize((-_mean / _std).tolist(), (1.0 / _std).tolist())


class ClassifierBenchmark:
    # load a model checkpoint and a test dataset
    # run a list of Metrics
    pass


class Metric(ABC):
    @abstractmethod
    def calculate(self, predictions: Tensor, labels: Tensor):
        pass


class ConfusionMatrixMetric(Metric):
    display: ConfusionMatrixDisplay

    def calculate(self, predictions: Tensor, labels: Tensor):
        predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
        targets = labels.cpu().numpy()

        self.display = ConfusionMatrixDisplay.from_predictions(targets, predicted_labels)

    def show(self):
        plt.figure()
        self.display.plot()
        plt.show()


def preprocess(data: TensorDataset) -> TensorDataset:
    img_data = data.tensors[0]

    if img_data.dtype == torch.uint8:
        img_data = img_data.float()
        img_data /= 255.0

    data.tensors = (normalize(img_data), data.tensors[1].long())

    return data


def show_image_with_label_and_prediction(image, label, prediction):
    plt.figure()
    predicted_label = torch.argmax(prediction)
    image = ToPILImage()(inverse_normalize(image).cpu())
    plt.imshow(np.asarray(image))
    plt.title(f'{label}, prediction: {predicted_label}')
    plt.show()


def get_free_memory():
    r = torch.cuda.memory_reserved(0)
    a = torch.cuda.memory_allocated(0)
    return r - a


def main():
    # should read number of classes from the model, since it might change
    # But it doesn't seem like there is a canonical way of getting that. Maybe just by passing in an input
    num_classes = 10
    batch_size = 64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # we need: train set, validation set, test set, model
    # image preprocessing
    # important: assert that the validation and test sets are identical to the split that was done when training
    # the classifier. Should add some kind of sanity check to ensure this
    # first steps, see available metrics in torch and calculate total and class-wise accuracy
    # train_data = CustomTensorDataset.load(Path(project_path('data/interim/gc10_train.pt')))
    test_dataset = CustomTensorDataset.load(Path(project_path('data/interim/gc10_test.pt')))
    # val_data = CustomTensorDataset.load(Path(project_path('data/interim/gc10_val.pt')))
    test_data = preprocess(test_dataset)
    # val_data = preprocess(val_data)

    if device == 'cuda:0':
        # if we get the Lightning classifier working instead of the old method,
        # we will need to implement the forward method
        classifier = torchvision.models.densenet201()
        classifier.classifier = nn.Sequential(
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
        checkpoint = torch.load(project_path('models/checkpoints/densenet201/epoch-18-val-0.111.pt'),
                                map_location=lambda storage, loc: storage)
        classifier.load_state_dict(checkpoint)
        classifier.eval()
        del checkpoint
        classifier.to(device)

    else:  # load to CPU instead
        classifier = DenseNet201.load_from_checkpoint(project_path('models/checkpoints/first_lightning_training.ckpt'),
                                                      map_location=torch.device('cpu'))

    # metric has the option 'average' with values micro, macro, and weighted. Might be worth looking at.
    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
    predictions = torch.empty((test_data.tensors[0].size()[0], num_classes), dtype=torch.float32)

    for i, batch in enumerate(tqdm((DataLoader(TensorDataset(test_data.tensors[0]), batch_size=batch_size)))):
        batch = batch.to(device)
        with torch.no_grad():
            predictions[i * batch_size:(i + 1) * batch_size] = classifier(batch)

    torch.save(predictions, project_path('data/interim/predictions_densenet.pt'))

    targets = test_data.tensors[1]

    print(f'smallest: {torch.min(targets)}, max: {torch.max(targets)}')

    # start, n = 20, 1

    # look at some random predictions
    # for image, label in zip(test_data.tensors[0][start:start + n + 1], test_data.tensors[1][start:start + n + 1]):
    #     prediction = classifier(image.unsqueeze(0))
    #     show_image_with_label_and_prediction(image, label, prediction)

    accuracy = metric(predictions, targets)
    print(accuracy)

    # use sklearn to create confusion matrix

    # OK, next steps in the analysis:
    # Create a modified Confusion Matrix where the secondary labels are considered
    # Qualitatively look at the misclassifications


def run_metrics_on_predictions_file():
    predictions = torch.load()


if __name__ == '__main__':
    main()
