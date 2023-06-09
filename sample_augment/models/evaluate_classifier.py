import inspect
import sys
from abc import abstractmethod, ABC
from copy import deepcopy, copy
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.cuda
import torchvision
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Normalize, ToPILImage
import torchvision.transforms as transforms
from tqdm import tqdm

from sample_augment import utils
from sample_augment.core import step, Artifact
from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.gc10.read_labels import GC10Labels
from sample_augment.data.train_test_split import stratified_split, stratified_k_fold_split, ValSet, \
    FoldDatasets
from sample_augment.models.train_classifier import TrainedClassifier, KFoldTrainedClassifiers, plain_transforms, \
    CustomViT
from sample_augment.utils import log, plot
from sample_augment.utils.plot import prepare_latex_plot

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


# def preprocess(data: AugmentDataset) -> AugmentDataset:
#     img_data = data.tensors[0]
#
#     if img_data.dtype == torch.uint8:
#         img_data = img_data.float()
#         img_data /= 255.0
#
#     data.tensors = (normalize(img_data), data.tensors[1].long())
#
#     return data


def show_image_with_label_and_prediction(image, label, prediction):
    plt.figure()
    predicted_label = torch.argmax(prediction)
    image = ToPILImage()(inverse_normalize(image).cpu())
    plt.imshow(np.asarray(image))
    plt.title(f'{label}, prediction: {predicted_label}')
    plt.show()


class ValidationPredictions(Artifact):
    predictions: Tensor


def get_preprocess(model):
    antialias_param_needed = 'antialias' in inspect.getfullargspec(transforms.RandomResizedCrop).args
    optional_aa_arg = {"antialias": True} if antialias_param_needed else {}

    _preprocess: List = copy(plain_transforms)
    if isinstance(model, CustomViT):
        _preprocess.insert(0, torchvision.transforms.Resize((224, 224), **optional_aa_arg))

    return transforms.Compose(_preprocess)


@step
def predict_validation_set(classifier: TrainedClassifier, validation_set: ValSet, batch_size: int) -> \
        ValidationPredictions:
    # should read number of classes from the model, since it might change
    # But it doesn't seem like there is a canonical way of getting that.
    # Maybe just by passing in an input
    assert hasattr(classifier.model, "num_classes"), "we need to have the num_classes attribute in our model"
    num_classes = classifier.model.num_classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    log.info(f"Prediction device: {device}")
    classifier.model.to(device)
    # deepcopy because consumed artifacts are not thrown away yet! (so state is mutable)
    val_data = deepcopy(validation_set)
    val_data.tensors = get_preprocess(classifier.model)(val_data.image_tensor), val_data.label_tensor
    # val_data = preprocess(val_data)

    # metric has the option 'average' with values micro, macro, and weighted.
    # Might be worth looking at.
    # metric = torchmetrics.classification.MulticlassAccuracy(num_classes=num_classes)
    predictions = torch.empty((val_data.tensors[0].size()[0], num_classes), dtype=torch.float32)

    for i, batch in enumerate(tqdm((DataLoader(TensorDataset(val_data.tensors[0]),
                                               batch_size=batch_size)),
                                   desc="Validation predictions", file=sys.stdout)):
        batch = batch[0].to(device)
        with torch.no_grad():
            predictions[i * batch_size:(i + 1) * batch_size] = classifier.model(batch)

    # torch.save(predictions, project_path('data/interim/predictions_densenet.pt'))
    # start, n = 20, 1

    # look at some random predictions
    # for image, label in zip(test_data.tensors[0][start:start + n + 1],
    #                         test_data.tensors[1][start:start + n + 1]):
    #     prediction = classifier(image.unsqueeze(0))
    #     show_image_with_label_and_prediction(image, label, prediction)

    # OK, next steps in the analysis:
    # Create a modified Confusion Matrix where the secondary labels are considered
    # Qualitatively look at the misclassifications

    return ValidationPredictions(predictions=predictions)


class ClassificationReport(Artifact):
    report: Dict


@step
def evaluate_classifier(val_pred_artifact: ValidationPredictions, val_set: ValSet,
                        gc10_labels: GC10Labels) -> ClassificationReport:
    # quickfix since the artifacts are not properly guarded from being mutated yet (TODO)
    # I removed the preprocessing of val_set since it's only used for the label info here, the data doesn't get accessed
    predictions = val_pred_artifact.predictions
    assert len(predictions) == len(val_set)
    imgs, labels = val_set.tensors[0], val_set.tensors[1]
    secondary_labels = gc10_labels.labels

    apply_secondary_labels(labels, predictions, secondary_labels, val_set)
    # ConfusionMatrixMetric(labels=classes).calculate(predictions, labels).show()
    # show_some_test_images(classes, imgs, labels, predictions, sec_labels, test_data)
    # debug_class_distribution(classes, labels, sec_labels)
    predicted_labels = torch.argmax(predictions, dim=1)
    report = classification_report(labels.numpy(), predicted_labels.numpy(),
                                   target_names=gc10_labels.class_names[:gc10_labels.number_of_classes],
                                   zero_division=0, output_dict=True)
    report_text = classification_report(labels.numpy(), predicted_labels.numpy(),
                                        target_names=gc10_labels.class_names[:gc10_labels.number_of_classes],
                                        zero_division=0, output_dict=False)
    log.info("\n" + report_text)

    return ClassificationReport(report=report)


class KFoldClassificationReport(Artifact):
    """one report for each fold"""
    reports: List[ClassificationReport]


@step
def evaluate_k_classifiers(dataset: AugmentDataset,
                           classifiers: KFoldTrainedClassifiers,
                           gc10_labels: GC10Labels,
                           test_ratio: float,
                           min_instances: int,
                           random_seed: int,
                           n_folds: int,
                           shared_directory: Path,
                           name: str
                           ) -> KFoldClassificationReport:
    # quick and dirty: for getting the splits these were trained on, take the same code
    train_val, _test = stratified_split(dataset, 1.0 - test_ratio, random_seed, min_instances)
    splits: FoldDatasets = stratified_k_fold_split(train_val, n_folds, random_seed, min_instances)

    metrics = []
    reports = []
    for classifier, (_train, val) in zip(classifiers.classifiers, splits.datasets):
        predictions: ValidationPredictions = predict_validation_set(classifier, ValSet.from_existing(val), 32)
        report: ClassificationReport = evaluate_classifier(predictions, ValSet.from_existing(val), gc10_labels)
        reports.append(report)
        metrics.append(report.report)

    # Compute the mean and standard deviation for each metric
    classes = gc10_labels.class_names[:gc10_labels.number_of_classes]
    metric_stats = {class_name: {} for class_name in classes}
    for class_name in classes:
        metric_stats[class_name]['precision'] = (np.mean([m[class_name]['precision'] for m in metrics]),
                                                 np.std([m[class_name]['precision'] for m in metrics]))
        metric_stats[class_name]['recall'] = (np.mean([m[class_name]['recall'] for m in metrics]),
                                              np.std([m[class_name]['recall'] for m in metrics]))
        metric_stats[class_name]['f1-score'] = (np.mean([m[class_name]['f1-score'] for m in metrics]),
                                                np.std([m[class_name]['f1-score'] for m in metrics]))
        metric_stats[class_name]['support'] = (np.mean([m[class_name]['support'] for m in metrics]),
                                               np.std([m[class_name]['support'] for m in metrics]))

    # Create a pandas DataFrame to display the results in a tabular format
    df = pd.DataFrame(columns=['Class', 'Precision', 'Recall', 'F1-Score', 'Support'])
    total_support = 0
    for class_name, stats in metric_stats.items():
        df_temp = pd.DataFrame({
            'Class': [class_name],
            'Precision': [f"{stats['precision'][0]:.2f} ± {stats['precision'][1]:.2f}"],
            'Recall': [f"{stats['recall'][0]:.2f} ± {stats['recall'][1]:.2f}"],
            'F1-Score': [f"{stats['f1-score'][0]:.2f} ± {stats['f1-score'][1]:.2f}"],
            'Support': [f"{int(stats['support'][0])}"]
        })
        total_support += int(stats['support'][0])

        df = pd.concat([df, df_temp], ignore_index=True)

    macro_precision = np.mean([metric['macro avg']['precision'] for metric in metrics])
    macro_precision_std = np.std([metric['macro avg']['precision'] for metric in metrics])
    macro_recall = np.mean([metric['macro avg']['recall'] for metric in metrics])
    macro_recall_std = np.std([metric['macro avg']['recall'] for metric in metrics])
    macro_f1 = np.mean([metric['macro avg']['f1-score'] for metric in metrics])
    macro_f1_std = np.std([metric['macro avg']['f1-score'] for metric in metrics])

    weighted_precision = np.mean([metric['weighted avg']['precision'] for metric in metrics])
    weighted_precision_std = np.std([metric['weighted avg']['precision'] for metric in metrics])
    weighted_recall = np.mean([metric['weighted avg']['recall'] for metric in metrics])
    weighted_recall_std = np.std([metric['weighted avg']['recall'] for metric in metrics])
    weighted_f1 = np.mean([metric['weighted avg']['f1-score'] for metric in metrics])
    weighted_f1_std = np.std([metric['weighted avg']['f1-score'] for metric in metrics])

    # add to dataframe
    df = pd.concat([
        df,
        pd.DataFrame({
            'Class': ['Macro average'],
            'Precision': [f"{macro_precision:.2f} ± {macro_precision_std:.2f}"],
            'Recall': [f"{macro_recall:.2f} ± {macro_recall_std:.2f}"],
            'F1-Score': [f"{macro_f1:.2f} ± {macro_f1_std:.2f}"],
            'Support': [total_support]
        })
    ], ignore_index=True)

    df = pd.concat([
        df,
        pd.DataFrame({
            'Class': ['Weighted average'],
            'Precision': [f"{weighted_precision:.2f} ± {weighted_precision_std:.2f}"],
            'Recall': [f"{weighted_recall:.2f} ± {weighted_recall_std:.2f}"],
            'F1-Score': [f"{weighted_f1:.2f} ± {weighted_f1_std:.2f}"],
            'Support': [total_support]
        })
    ], ignore_index=True)
    print(df)

    df.to_csv(shared_directory / f'classifier_report_{name}.csv', index=False)

    return KFoldClassificationReport(reports=reports)


@step
def k_fold_plot_loss_over_epochs(classifiers: KFoldTrainedClassifiers, shared_directory: Path, name: str):
    train_losses = []
    validation_losses = []

    for classifier in classifiers.classifiers:
        train_losses.append(classifier.metrics.train_loss)
        validation_losses.append(classifier.metrics.validation_loss)

    # Convert to numpy arrays
    train_losses = np.array(train_losses)
    validation_losses = np.array(validation_losses)

    # Compute mean and standard deviation
    train_mean = np.mean(train_losses, axis=0)
    validation_mean = np.mean(validation_losses, axis=0)
    train_std = np.std(train_losses, axis=0)
    validation_std = np.std(validation_losses, axis=0)

    # Number of epochs
    epochs = range(1, len(train_mean) + 1)

    prepare_latex_plot()
    # Create the plot
    plt.figure(figsize=(utils.plot.text_width, 7.0 / 10.0 * utils.plot.text_width))
    x_ticks = np.arange(1, max(epochs) + 1, 5)
    x_ticks = x_ticks - 1
    x_ticks = np.insert(x_ticks, 1, 1)  # Insert 1 at the beginning
    # Training and validation loss for each classifier in light color
    for i in range(train_losses.shape[0]):
        plt.plot(epochs, train_losses[i], color='orange', alpha=0.15)
        plt.plot(epochs, validation_losses[i], color='blue', alpha=0.1)

    # Average training loss
    plt.plot(epochs, train_mean, label='Mean Training Loss', color='orange', linewidth=2)
    plt.fill_between(epochs, train_mean - train_std, train_mean + train_std, color='orange', alpha=0.25)

    # Average validation loss
    plt.plot(epochs, validation_mean, label='Mean Validation Loss', color='blue', linewidth=2)
    plt.fill_between(epochs, validation_mean - validation_std, validation_mean + validation_std, color='blue',
                     alpha=0.15)

    # Labels, title and legend
    # plt.title('Average Cross-Entropy Loss per Epoch with Standard Deviation')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()
    plt.xlim([1, len(train_mean)])
    plt.ylim([0.0, 2.25])
    y_ticks, _ = plt.yticks()
    y_ticks = y_ticks[1:]
    plt.yticks(y_ticks)
    plt.xticks(x_ticks)

    # dirty nested f-string - because I can
    log.info(f"Saving KFold-Losses plot to {shared_directory / f'kfold_losses_{name}.pgf'}")
    plt.tight_layout()

    plt.savefig(shared_directory / f"kfold_losses_{name}.pgf", bbox_inches='tight', backend="pgf")
    plt.savefig(shared_directory / f"kfold_losses_{name}.pdf", bbox_inches='tight', backend="pgf")


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
    log.info(f'accuracy: {1.0 - len(misclassification_idx) / len(predictions):.2f}')
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
def plot_loss_over_epochs(classifier: TrainedClassifier, shared_directory: Path):
    # log.info(f"Training:   {classifier.metrics.train_loss}")
    # log.info(f"Validation: {classifier.metrics.validation_loss}")
    plt.figure(figsize=(10, 7))

    plt.plot(classifier.metrics.train_loss, label='Training Loss', color='blue')
    plt.plot(classifier.metrics.validation_loss, label='Validation Loss', color='red')

    plt.title('Cross-Entropy Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()

    plt.savefig(shared_directory / "losses.png")


if __name__ == '__main__':
    evaluate_classifier()
