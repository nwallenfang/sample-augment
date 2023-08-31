import inspect
import sys
from copy import deepcopy, copy
from pathlib import Path
from pprint import pprint
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch.cuda
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import classification_report
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torchvision.transforms import Normalize, ToPILImage
from tqdm import tqdm

from sample_augment.core import step, Artifact
from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.gc10.read_labels import GC10Labels
from sample_augment.data.train_test_split import stratified_split, stratified_k_fold_split, ValSet, \
    FoldDatasets
from sample_augment.models.classifier import VisionTransformer
from sample_augment.models.train_classifier import TrainedClassifier, KFoldTrainedClassifiers, plain_transforms, \
    ClassifierMetrics
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir
from sample_augment.utils.plot import prepare_latex_plot

_mean = torch.tensor([0.485, 0.456, 0.406])
_std = torch.tensor([0.229, 0.224, 0.225])
normalize = Normalize(mean=_mean, std=_std)
# reverse operation for use in visualization
inverse_normalize = Normalize((-_mean / _std).tolist(), (1.0 / _std).tolist())


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
    if isinstance(model, VisionTransformer):
        # VisionTransformer needs to do resizing
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
            predictions[i * batch_size:(i + 1) * batch_size] = torch.sigmoid(classifier.model(batch))

    return ValidationPredictions(predictions=predictions)


class ClassificationReport(Artifact):
    report: Dict


def show_prediction(image, prediction, class_names, true_labels, name: str, img_id: str, threshold=0.5):
    # Assume image is torch.Tensor, prediction is torch.Tensor, class_names is list of strings,
    # true_labels is list of strings
    prediction = prediction.cpu().numpy()
    image = image.cpu().numpy()
    image = np.transpose(image, (1, 2, 0))  # assuming image tensor is (C, H, W)

    prepare_latex_plot()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    # Show image
    ax[0].imshow(image)
    ax[0].axis('off')

    # Show prediction
    colors = ['blue' if class_name in true_labels else 'orange' for class_name in class_names]
    ax[1].axhline(y=threshold, color='r', linestyle='--')
    sns.barplot(x=class_names, y=prediction, palette=colors, ax=ax[1])
    ax[1].set_ylabel('Prediction')
    ax[1].tick_params(axis='x', rotation=90)  # rotate x-axis labels for better visibility

    plt.tight_layout()
    plt.savefig(shared_dir / "figures" / "predictions" / f"{name}_{img_id}.pdf", bbox_inches="tight")


@step
def evaluate_classifier(val_pred_artifact: ValidationPredictions, val_set: ValSet,
                        gc10_labels: GC10Labels, threshold_lambda: float) -> ClassificationReport:
    from sample_augment.models.train_classifier import determine_threshold_vector
    # quickfix since the artifacts are not properly guarded from being mutated yet (TODO)
    # I removed the preprocessing of val_set since it's only used for the label info here, the data doesn't get accessed
    predictions = val_pred_artifact.predictions
    assert len(predictions) == len(val_set)
    imgs, labels = val_set.tensors[0], val_set.tensors[1]

    threshold_vector = determine_threshold_vector(predictions, val_set, threshold_lambda)
    log.info(f'Threshold vector: {threshold_vector}')

    # for i in range(10):
    #     true_class_names = [gc10_labels.class_names[j] for j in labels[i].nonzero().flatten().tolist()]
    #     show_prediction(imgs[i], torch.sigmoid(predictions[i]), gc10_labels.class_names, true_labels=true_class_names,
    #                     name=val_pred_artifact.configs['name'], img_id=val_set.img_ids[i], threshold=threshold)

    # apply_secondary_labels(labels, predictions, secondary_labels, val_set)
    # ConfusionMatrixMetric(labels=classes).calculate(predictions, labels).show()
    # show_some_test_images(classes, imgs, labels, predictions, sec_labels, test_data)
    # debug_class_distribution(classes, labels, sec_labels)
    predicted_labels = (predictions > threshold_vector.unsqueeze(0)).float()
    report = classification_report(labels.numpy(), predicted_labels.numpy(),
                                   target_names=gc10_labels.class_names[:gc10_labels.number_of_classes],
                                   zero_division=0, output_dict=True)
    report_text = classification_report(labels.numpy(), predicted_labels.numpy(),
                                        target_names=gc10_labels.class_names[:gc10_labels.number_of_classes],
                                        zero_division=0, output_dict=False)
    log.info("\n" + report_text)

    return ClassificationReport(report=report)


class KFoldClassificationReport(Artifact):
    """one report for each fold, report is a dict with the report output from scikit-learn"""
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
                           name: str,
                           threshold_lambda: float
                           ) -> KFoldClassificationReport:
    # quick and dirty: for getting the splits these were trained on, take the same code
    train_val, _test = stratified_split(dataset, 1.0 - test_ratio, random_seed, min_instances)
    splits: FoldDatasets = stratified_k_fold_split(train_val, n_folds, random_seed, min_instances)

    metrics = []
    reports = []
    for classifier, (_train, val) in zip(classifiers.classifiers, splits.datasets):
        predictions: ValidationPredictions = predict_validation_set(classifier, ValSet.from_existing(val), 32)
        report: ClassificationReport = evaluate_classifier(predictions, ValSet.from_existing(val), gc10_labels,
                                                           threshold_lambda)
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


def k_fold_plot_loss_over_epochs(metrics_by_run: Dict[str, List[ClassifierMetrics]], figure_dir: Path, name: str,
                                 ax=None, ylim=None, yticks=None, color_dict=None):
    data = []
    for run_name, metrics in metrics_by_run.items():
        for fold, metric in enumerate(metrics):
            for epoch, (train_loss, val_loss) in enumerate(zip(metric.train_loss, metric.validation_loss)):
                data.append({
                    'Run': run_name,
                    'Fold': fold,
                    'Epoch': epoch + 1,  # 1-indexing for epochs
                    'Training': train_loss,
                    'Validation': val_loss
                })

    df = pd.DataFrame(data)
    # Reshaping the DataFrame
    df_melt = df.melt(id_vars=['Run', 'Fold', 'Epoch'],
                      value_vars=['Training', 'Validation'],
                      var_name='Loss Type',
                      value_name='Loss')

    # Create color palette for each run and loss type combination
    unique_runs = df_melt['Run'].unique()
    colors = sns.color_palette("tab10", len(unique_runs))
    if not color_dict:
        color_dict = {run: color for run, color in zip(unique_runs, colors)}

    # Generate plot
    if ax is None:
        prepare_latex_plot()
        fig, ax = plt.subplots(figsize=(8, 4))

    sns.lineplot(ax=ax if ax else plt.gca(), data=df_melt, x='Epoch', y='Loss',
                 hue='Run', style='Loss Type',
                 palette=color_dict, errorbar='sd')

    if ylim:
        ax.set_ylim(ylim)
    if yticks:
        ax.set_yticks(yticks)

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Binary Cross-Entropy Loss')
    # plt.legend(handles=handles[1:], labels=labels[1:], title='Run',
    #            loc='upper right')
    # plt.legend(handles=handles[1:], labels=labels[1:], title='Run and Loss Type', bbox_to_anchor=(1.05, 1),
    #            loc='upper left')
    plt.tight_layout()
    # idk about this but fine huh
    # plt.savefig(figure_dir / f"kfold_losses_{name}.pdf", bbox_inches='tight')


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
def plot_loss_over_epochs(metrics: ClassifierMetrics):
    # log.info(f"Training:   {classifier.metrics.train_loss}")
    # log.info(f"Validation: {classifier.metrics.validation_loss}")
    prepare_latex_plot()
    plt.figure(figsize=(8, 5))

    plt.plot(metrics.train_loss, label='Training Loss', color='blue')
    plt.plot(metrics.validation_loss, label='Validation Loss', color='red')

    plt.title('Cross-Entropy Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Cross-Entropy Loss')
    plt.legend()

    plt.savefig(shared_dir / "figures" / f"{metrics.configs['name']}_losses.pdf")


if __name__ == '__main__':
    evaluate_classifier()


def plot_roc_curves(fpr: Dict[int, np.ndarray],
                    tpr: Dict[int, np.ndarray],
                    roc_auc: Dict[int, float]) -> None:
    # TODO wire this one up
    """Plot ROC curves for each class."""
    plt.figure(figsize=(10, 8))

    for i in range(len(fpr)):
        plt.plot(fpr[i], tpr[i],
                 label='ROC curve of class {0} (area = {1:0.2f})'
                       ''.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([-0.05, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic to Multi-Class')
    plt.legend(loc="lower right")
    plt.show()

# def calculate_optimal_thresholds(true_labels: np.ndarray,
#                                  probabilities: np.ndarray) -> Dict[int, float]:
#     """Calculate optimal threshold for each class using Youden's J statistic."""
#     # Ensure the true labels are binary
#     if len(true_labels.shape) > 1 and true_labels.shape[1] > 1:
#         binary_labels = true_labels
#     else:
#         binary_labels = label_binarize(true_labels, classes=np.unique(true_labels))
#
#     opt_threshold_dict = dict()
#
#     # Compute optimal threshold for each class
#     for i in range(binary_labels.shape[1]):
#         fpr, tpr, thresholds = roc_curve(binary_labels[:, i], probabilities[:, i])
#         j_scores = tpr - fpr
#         opt_idx = np.argmax(j_scores)
#         opt_threshold_dict[i] = thresholds[opt_idx]
#
#     return opt_threshold_dict
