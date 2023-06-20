import sys
from abc import abstractmethod, ABC
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
import torchmetrics
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Normalize, ToPILImage
from tqdm import tqdm

from sample_augment.core import step, Artifact
from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.gc10.read_labels import GC10Labels
from sample_augment.data.train_test_split import TestSet
from sample_augment.models.train_classifier import TrainedClassifier, CustomDenseNet

_mean = torch.tensor([0.485, 0.456, 0.406])
_std = torch.tensor([0.229, 0.224, 0.225])
normalize = Normalize(mean=_mean, std=_std)
# reverse operation for use in visualization
inverse_normalize = Normalize((-_mean / _std).tolist(), (1.0 / _std).tolist())


class Metric(ABC):
    @abstractmethod
    def calculate(self, predictions: Tensor, labels: Tensor):
        pass


class ConfusionMatrixMetric(Metric):
    display: ConfusionMatrixDisplay
    labels: str

    def __init__(self, labels=None):
        if labels:
            self.labels = labels

    def calculate(self, predictions: Tensor, labels: Tensor) -> "ConfusionMatrixMetric":
        predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
        targets = labels.cpu().numpy()

        self.display = ConfusionMatrixDisplay.from_predictions(targets, predicted_labels,
                                                               display_labels=self.labels)
        return self

    @staticmethod
    def show(title=None):
        if title:
            plt.suptitle(title)
        plt.show()


def preprocess(data: AugmentDataset) -> AugmentDataset:
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


class TestPredictions(Artifact):
    predictions: Tensor


@step
def predict_testset(classifier: TrainedClassifier, test_dataset: TestSet, batch_size: int) -> TestPredictions:
    # should read number of classes from the model, since it might change
    # But it doesn't seem like there is a canonical way of getting that.
    # Maybe just by passing in an input
    assert isinstance(classifier.model, CustomDenseNet), "only support DenseNet for now , we need to read " \
                                                         "num_classes"
    num_classes = classifier.model.num_classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # image preprocessing
    # important: assert that the validation and test sets are identical to the split that was
    # done when training the classifier. Should add some kind of sanity check to ensure this
    # first steps, see available metrics in torch and calculate total and class-wise accuracy

    test_data = preprocess(test_dataset)
    # val_data = preprocess(val_data)

    # metric has the option 'average' with values micro, macro, and weighted.
    # Might be worth looking at.
    metric = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
    predictions = torch.empty((test_data.tensors[0].size()[0], num_classes), dtype=torch.float32)

    for i, batch in enumerate(tqdm((DataLoader(TensorDataset(test_data.tensors[0]),
                                               batch_size=batch_size)),
                                   desc="Test predictions", file=sys.stdout)):
        batch = batch[0].to(device)
        with torch.no_grad():
            predictions[i * batch_size:(i + 1) * batch_size] = classifier.model(batch)

    # torch.save(predictions, project_path('data/interim/predictions_densenet.pt'))

    targets = test_data.tensors[1]

    print(f'smallest: {torch.min(targets)}, max: {torch.max(targets)}')

    # start, n = 20, 1

    # look at some random predictions
    # for image, label in zip(test_data.tensors[0][start:start + n + 1],
    #                         test_data.tensors[1][start:start + n + 1]):
    #     prediction = classifier(image.unsqueeze(0))
    #     show_image_with_label_and_prediction(image, label, prediction)

    accuracy = metric(predictions, targets)
    print(accuracy)

    # use sklearn to create confusion matrix

    # OK, next steps in the analysis:
    # Create a modified Confusion Matrix where the secondary labels are considered
    # Qualitatively look at the misclassifications

    return TestPredictions(predictions=predictions)


@step
def run_metrics_on_predictions_file(test_pred: TestPredictions, test_data: TestSet,
                                    labels_artifact: GC10Labels):
    classes = [
        "punching_hole",
        "welding_line",
        "crescent_gap",
        "water_spot",
        "oil_spot",
        "silk_spot",
        "inclusion",
        "rolled_pit",
        "crease",
        "waist_folding"
    ]
    test_data = preprocess(test_data)
    predictions = test_pred.predictions

    # predictions = torch.load(project_path('data/ds_data/predictions_densenet.pt'))
    assert len(predictions) == len(test_data)

    imgs, labels = test_data.tensors[0], test_data.tensors[1]
    # ConfusionMatrixMetric().calculate(predictions, labels).show()

    # retrieve secondary labels for test instances
    # with open(project_path('data/interim/labels.json', 'r')) as label_json_file:
    #     label_info = json.load(label_json_file)
    sec_labels = labels_artifact.labels
    for i in range(10):
        i += 10
        test_img = imgs[i]
        test_img_id = test_data.img_ids[i]
        test_img_path = test_data.root_dir / str(labels[i] + 1) / test_img_id
        test_img = ToPILImage()(inverse_normalize(test_img))
        secondary = [classes[sec] for sec in sec_labels[test_img_id]['secondary']]

        plt.imshow(test_img)
        plt.title(f'{test_img_id} - {test_img_path}')
        plt.figtext(0.5, 0.05, f'true: {classes[labels[i]]}, secondary: {secondary} '
                               f'- predicted: {classes[torch.argmax(predictions[i])]}',
                               ha='center', fontsize=9)
        plt.show()

    # count number of misclassifications
    # calc ratio of predicted labels that are part of secondary labels
    misclassification_idx = [i for i in range(len(predictions)) if
                             torch.argmax(predictions[i]) != labels[i]]

    print(f'test size: {len(predictions)}')
    print(f'number of misses: {len(misclassification_idx)}')
    number_of_secondary_hits = 0

    # let's be lenient towards the model and change all misses
    # with secondary hits to their secondary labels
    for idx in misclassification_idx:
        predicted_label = torch.argmax(predictions[idx])
        true_label = labels[idx]
        assert predicted_label != true_label
        secondary = sec_labels[test_data.img_ids[idx]]['secondary']

        # secondary can be a single class index or a list of indices
        if secondary and (predicted_label == secondary or predicted_label in secondary):
            labels[idx] = predicted_label
            number_of_secondary_hits += 1

    print(f'number of secondary hits: {number_of_secondary_hits}')
    ConfusionMatrixMetric(labels=classes).calculate(predictions, labels).show()
    confusion_mat = confusion_matrix(torch.argmax(predictions, dim=1).numpy(), labels.numpy())
    counts = np.bincount(labels)
    print(confusion_mat)
    print(' --- ')
    # TODO something is wrong here! The counts show that the classes aren't properly balanced
    #  need to see if this problem is already present in the train_test split
    # TODO verify that train, test and val sets are disjoint!!
    print(counts)


@step
def evaluate_classifier_old():
    # TODO integrate old main_method from this file
    pass


@step
def plot_loss_over_epochs(classifier: TrainedClassifier, figure_directory: Path):
    # log.info(f"Training:   {classifier.metrics.train_loss}")
    # log.info(f"Validation: {classifier.metrics.validation_loss}")
    plt.figure(figsize=(10, 7))

    plt.plot(classifier.metrics.train_loss, label='Training Loss', color='blue')
    plt.plot(classifier.metrics.validation_loss, label='Validation Loss', color='red')

    plt.title('Cross-Entropy Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()

    plt.savefig(figure_directory / "losses.png")


if __name__ == '__main__':
    run_metrics_on_predictions_file()
    # main()
