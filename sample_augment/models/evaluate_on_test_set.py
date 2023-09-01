import sys
from copy import deepcopy

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sample_augment.core import step, Artifact
from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.train_test_split import TestSet, create_train_test_val
from sample_augment.models.evaluate_classifier import get_preprocess
from sample_augment.models.train_classifier import TrainedClassifier
from sample_augment.utils import log


class TestSetPredictions(Artifact):
    serialize_this = False
    predictions: Tensor


def predict_test_set(classifier: TrainedClassifier, test_set: TestSet, batch_size: int = 32) -> \
        TestSetPredictions:
    """
        copied from predict_validation_set :)
    """
    assert hasattr(classifier.model, "num_classes"), "we need to have the num_classes attribute in our model"
    num_classes = classifier.model.num_classes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    classifier.model.to(device)
    # deepcopy because consumed artifacts are not thrown away yet! (so state is mutable)
    val_data = deepcopy(test_set)
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

    return TestSetPredictions(predictions=predictions)


class TestMetrics(Artifact):
    f1: float
    f1_std: float
    auc: float
    auc_std: float


@step
def f1_and_auc_on_test_set(test_set: TestSet, test_predictions: TestSetPredictions, bootstrap_m: int = 5000):
    # TODO try increasing M
    predictions = test_predictions.predictions.cpu().numpy()
    test_labels = test_set.label_tensor.cpu().numpy()

    f1_scores = []
    auc_scores = []

    indices = np.arange(len(test_labels)) 

    log.info(f'Doing {bootstrap_m} Bootstrap iterations..')
    # Bootstrapping loop
    for i in range(bootstrap_m):
        if i % 1000 == 0:
            log.info(f'Step {i}..')
        # Sample with replacement from original indices
        sample_indices = np.random.choice(indices, size=len(indices), replace=True)

        # Create the sample
        sample_true = test_labels[sample_indices]
        sample_pred = predictions[sample_indices]

        # Compute the metrics for the sample
        # TODO THRESHOLDING!! -> probably pass validation set
        f1_scores.append(f1_score(sample_true, np.round(sample_pred), average='macro'))
        auc_scores.append(roc_auc_score(sample_true, sample_pred, multi_class='ovr', average='macro'))

    # Compute mean and standard deviation
    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)
    auc_mean = np.mean(auc_scores)
    auc_std = np.std(auc_scores)

    print(f'Macro F1: {f1_mean:.3f}±{f1_std:.3f}, AUC: {auc_mean:.3f}±{auc_std:.3f}')
    return TestMetrics(f1=f1_mean, f1_std=f1_std, auc=auc_mean, auc_std=auc_std)


def main():
    # take a random one that's been just trained for 1 epoch to test this
    trained_classifier = TrainedClassifier.from_dict({
        "configs": {
            "lr_step_size": 10,
            "lr_gamma": 0.7,
            "model_type": "VisionTransformer",
            "num_epochs": 1,
            "batch_size": 32,
            "learning_rate": 0.0001,
            "balance_classes": True,
            "random_seed": 100,
            "data_augment": True,
            "geometric_augment": True,
            "color_jitter": 0.25,
            "h_flip_p": 0.5,
            "v_flip_p": 0.5,
            "lr_schedule": False,
            "threshold_lambda": 0.4,
            "name": "s00-baseline",
            "test_ratio": 0.2,
            "val_ratio": 0.1,
            "min_instances": 10
        },
        "model": {
            "type": "torch.nn.Module",
            "class": "sample_augment.models.classifier.VisionTransformer",
            "kwargs": {
                "num_classes": 10
            },
            "path": "TrainedClassifier/27f9e5_model.pt"
        },
        "metrics": {
            "configs": {},
            "train_loss": {
                "type": "numpy.ndarray",
                "path": "ClassifierMetrics/27f9e5_train_loss.npy"
            },
            "validation_loss": {
                "type": "numpy.ndarray",
                "path": "ClassifierMetrics/27f9e5_validation_loss.npy"
            },
            "train_accuracy": {
                "type": "numpy.ndarray",
                "path": "ClassifierMetrics/27f9e5_train_accuracy.npy"
            },
            "validation_accuracy": {
                "type": "numpy.ndarray",
                "path": "ClassifierMetrics/27f9e5_validation_accuracy.npy"
            },
            "validation_f1": {
                "type": "numpy.ndarray",
                "path": "ClassifierMetrics/27f9e5_validation_f1.npy"
            },
            "epoch": 1
        }})
    complete_dataset = AugmentDataset.from_name('dataset_f00581')
    bundle = create_train_test_val(complete_dataset, random_seed=100, test_ratio=0.2, val_ratio=0.1, min_instances=10)
    test_predictions = predict_test_set(trained_classifier, bundle.test)
    f1_and_auc_on_test_set(bundle.test, test_predictions)


if __name__ == '__main__':
    main()
