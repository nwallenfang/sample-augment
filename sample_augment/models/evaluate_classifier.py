import sys
from abc import abstractmethod, ABC
from copy import deepcopy
from pathlib import Path
from pprint import pprint
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch.cuda
import torchmetrics
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Normalize, ToPILImage
from tqdm import tqdm

from sample_augment.core import step, Artifact
from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.gc10.read_labels import GC10Labels
from sample_augment.data.train_test_split import TestSet
from sample_augment.models.train_classifier import TrainedClassifier, CustomDenseNet, KFoldTrainedClassifiers
from sample_augment.utils import log

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

    # deepcopy because consumed artifacts are not thrown away yet! (so state is mutable)
    test_data = preprocess(deepcopy(test_dataset))
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


class ClassificationF1Report(Artifact):
    report: Dict


@step
def evaluate_classifier(test_pred: TestPredictions, test_data: TestSet,
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
    # quickfix since the artifacts are not properly guarded from being mutated yet (TODO)
    test_data = preprocess(deepcopy(test_data))
    predictions = test_pred.predictions
    assert len(predictions) == len(test_data)
    imgs, labels = test_data.tensors[0], test_data.tensors[1]
    sec_labels = labels_artifact.labels
    # show_some_test_images(classes, imgs, labels, predictions, sec_labels, test_data)

    # count number of misclassifications
    # calc ratio of predicted labels that are part of secondary labels
    apply_secondary_labels(labels, predictions, sec_labels, test_data)
    # ConfusionMatrixMetric(labels=classes).calculate(predictions, labels).show()

    # debug_class_distribution(classes, labels, sec_labels)
    predicted_labels = torch.argmax(predictions, dim=1)
    report = classification_report(labels.numpy(), predicted_labels.numpy(), target_names=classes,
                                   zero_division=0, output_dict=True)
    report_text = classification_report(labels.numpy(), predicted_labels.numpy(), target_names=classes,
                                        zero_division=0, output_dict=False)
    print(report_text)
    for class_name in classes:
        print(f'Accuracy for class {class_name}: {report[class_name]}')

    return ClassificationF1Report(report=report)


@step
def evaluate_k_classifiers(classifiers: KFoldTrainedClassifiers):
    print(classifiers.classifiers[0].metrics)


def show_some_test_images(classes, imgs, labels, predictions, sec_labels, test_data):
    for i in range(10):
        i += 10
        test_img = imgs[i]
        test_img_id = test_data.img_ids[i]
        test_img_path = test_data.root_dir / str(labels[i].item() + 1) / test_img_id
        test_img = ToPILImage()(inverse_normalize(test_img))
        # test_img = ToPILImage()(test_img)
        secondary = [classes[sec] for sec in sec_labels[test_img_id]['secondary']]

        plt.imshow(test_img)
        plt.title(f'{test_img_id} - {test_img_path}')
        plt.figtext(0.5, 0.05, f'true: {classes[labels[i]]}, secondary: {secondary} '
                               f'- predicted: {classes[torch.argmax(predictions[i])]}',
                    ha='center', fontsize=9)
        plt.show()


def debug_class_distribution(classes, labels, sec_labels):
    log.debug(' --- ')
    # class counts for test set
    counts = np.bincount(labels)
    ratios = counts / len(labels)
    ratio_dict = {classes[i]: f"{ratios[i] * 100:.2f}%" for i in range(len(ratios))}
    # log.info(f"TestSet: {pprint(ratio_dict)}")
    print("test_set:")
    pprint(ratio_dict)
    total_labels = [label['y'] - 1 for label in sec_labels.values()]
    total_counts = np.bincount(total_labels)
    total_ratios = total_counts / len(total_labels)
    total_ratio_dict = {classes[i]: f"{total_ratios[i] * 100:.2f}%" for i in range(len(total_ratios))}
    print("total:")
    pprint(total_ratio_dict)


def apply_secondary_labels(labels, predictions, sec_labels, test_data):
    misclassification_idx = [i for i in range(len(predictions)) if
                             torch.argmax(predictions[i]) != labels[i]]
    log.debug(f'test size: {len(predictions)}')
    log.info(f'accuracy: {1.0 - len(misclassification_idx) / len(predictions)}')
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
    log.debug(f'number of secondary hits: {number_of_secondary_hits}')


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
    evaluate_classifier()
