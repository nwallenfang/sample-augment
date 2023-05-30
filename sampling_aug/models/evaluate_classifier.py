from abc import abstractmethod, ABC
from pathlib import Path

import torch.cuda
import torchmetrics
from torch.utils.data import TensorDataset
from torchvision.transforms import Normalize, ToPILImage

from data.dataset import CustomTensorDataset
from models.DenseNetClassifier import DenseNet201
from utils.paths import project_path

import matplotlib.pyplot as plt
import numpy as np

from sklearn.metrics import ConfusionMatrixDisplay

_mean = torch.tensor([0.485, 0.456, 0.406])
_std = torch.tensor([0.229, 0.224, 0.225])
normalize = Normalize(mean=_mean, std=_std)
# reverse operation for use in visualization
inverse_normalize = Normalize((-_mean / _std).tolist(), (1.0 / _std).tolist())


class Metric(ABC):
    @abstractmethod
    def show(self, **kwargs):
        pass


class ConfusionMatrixMetric(Metric):
    def show(self, **kwargs):
        pass

    test = 10


def preprocess(data: TensorDataset) -> TensorDataset:
    # TODO if dtype == uint8
    img_data = data.tensors[0]

    # plt.imshow(img_data[5].permute(1, 2, 0))
    # plt.show()

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


def main():
    num_classes = 10
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
    # TODO might do random shuffling of test set, doesn't affect the metrics but the test set is ordered by class
    #  currently

    del test_dataset
    # val_data = preprocess(val_data)
    if torch.cuda.is_available():
        classifier = DenseNet201.load_from_checkpoint(project_path('models/checkpoints/first_lightning_training.ckpt'))
        # move data to GPU
        test_data.tensors = (test_data.tensors[0].to(device), test_data.tensors[1].to(device))
    else:  # load to CPU
        classifier = DenseNet201.load_from_checkpoint(project_path('models/checkpoints/first_lightning_training.ckpt'),
                                                      map_location=torch.device('cpu'))

    # TODO read number of classes from the model, since it might change
    # metric has the option 'average' with values micro, macro, and weighted. Might be worth looking at.
    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes).to(device)

    predictions = classifier.model(test_data.tensors[0])

    targets = test_data.tensors[1]

    print(f'smallest: {torch.min(targets)}, max: {torch.max(targets)}')

    start, n = 20, 1

    # look at some random predictions
    for image, label in zip(test_data.tensors[0][start:start + n + 1], test_data.tensors[1][start:start + n + 1]):
        prediction = classifier.model(image.unsqueeze(0))
        show_image_with_label_and_prediction(image, label, prediction)

    accuracy = metric(predictions, targets)
    # print(accuracy)    

    # use sklearn to create confusion matrix
    predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
    labels = targets.cpu().numpy()

    display = ConfusionMatrixDisplay.from_predictions(labels, predicted_labels)
    plt.figure()
    display.plot()
    plt.show()


if __name__ == '__main__':
    main()
