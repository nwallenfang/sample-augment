import sys
from copy import deepcopy
from typing import List, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, roc_auc_score, classification_report
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from sample_augment.core import Artifact
from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.train_test_split import TestSet, create_train_test_val, TrainTestValBundle, ValSet
from sample_augment.models.evaluate_classifier import get_preprocess, predict_validation_set, ValidationPredictions
from sample_augment.models.generator import GC10_CLASSES
from sample_augment.models.train_classifier import TrainedClassifier, determine_threshold_vector
from sample_augment.sampling.compare_strategies import MultiSeedStrategyComparison
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir
from sample_augment.utils.plot import prepare_latex_plot


class TestSetPredictions(Artifact):
    serialize_this = True
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


def f1_and_auc_on_test_set(test_labels: Tensor, threshold_vec: Tensor, test_predictions: Tensor,
                           bootstrap_m: int = 5000) -> TestMetrics:
    predictions = test_predictions.cpu().numpy()
    f1_scores = []
    auc_scores = []
    threshold_vec = threshold_vec.cpu().numpy()
    indices = np.arange(len(test_labels))

    log.debug(f'Doing {bootstrap_m} Bootstrap iterations..')
    # Bootstrapping loop
    for i in range(bootstrap_m):
        if i % 1000 == 0:
            log.debug(f'Step {i}..')
        # Sample with replacement from original indices
        sample_indices = np.random.choice(indices, size=len(indices), replace=True)

        # Create the sample
        sample_true = test_labels[sample_indices]
        sample_pred = predictions[sample_indices]

        # our custom thresholding - verify this is correct
        f1_scores.append(f1_score(sample_true, sample_pred > threshold_vec, average='macro'))
        auc_scores.append(roc_auc_score(sample_true, sample_pred, multi_class='ovr', average='macro'))

    # Compute mean and standard deviation
    f1_mean = np.mean(f1_scores)
    f1_std = np.std(f1_scores)
    auc_mean = np.mean(auc_scores)
    auc_std = np.std(auc_scores)

    log.debug(f'Macro F1: {f1_mean:.3f}±{f1_std:.3f}, AUC: {auc_mean:.3f}±{auc_std:.3f}')
    return TestMetrics(f1=f1_mean, f1_std=f1_std, auc=auc_mean, auc_std=auc_std)


def read_baseline_classifiers() -> List[TrainedClassifier]:
    comparison = MultiSeedStrategyComparison.from_name('s01-baseline_df5f64.json')
    baselines = [comp.baseline for comp in comparison.strategy_comparisons]
    return baselines


def read_synth_classifiers() -> List[TrainedClassifier]:
    comparison = MultiSeedStrategyComparison.from_name('s16-unified-generator_03aa95.json')
    # strat_idx = comparison.configs['strategies'].index('')
    classifiers = [comp.classifiers[0] for comp in comparison.strategy_comparisons]
    return classifiers


def eval_single_classifier(classifier: TrainedClassifier,
                           bundle: TrainTestValBundle,
                           name: str) -> Tuple[TestSetPredictions, TestMetrics, Tensor, ValidationPredictions]:
    val_predictions = predict_validation_set(classifier, bundle.val, batch_size=32)
    threshold_vec = determine_threshold_vector(val_predictions.predictions, bundle.val, threshold_lambda=0.4)
    test_labels = bundle.test.label_tensor.cpu().numpy()
    test_predictions = predict_test_set(classifier, bundle.test)
    test_predictions.configs['name'] = name
    metrics = f1_and_auc_on_test_set(test_labels, threshold_vec, test_predictions.predictions, bootstrap_m=1)
    return test_predictions, metrics, threshold_vec, val_predictions


def main():
    complete_dataset = AugmentDataset.from_name('dataset_f00581')
    bundle = create_train_test_val(complete_dataset, random_seed=100, test_ratio=0.2, val_ratio=0.1, min_instances=10)
    log.info('reading baseline..')
    baseline_classifiers = read_baseline_classifiers()
    log.info('reading synth..')
    synth_classifiers = read_synth_classifiers()

    log.info('eval baseline..')
    baseline = [eval_single_classifier(clas, bundle, name=f'baseline_{i}') for i, clas in
                enumerate(baseline_classifiers)]
    baseline_predictions, baseline_metrics, baseline_thresholds, baseline_val = zip(*baseline)
    baseline_thresholds = torch.stack(baseline_thresholds)
    baseline_val = torch.stack([val.predictions for val in baseline_val])
    log.info('eval synth..')
    synth = [eval_single_classifier(clas, bundle, name=f'synth_{i}') for i, clas in enumerate(synth_classifiers)]
    synth_predictions, synth_metrics, synth_thresholds, synth_val = zip(*synth)
    synth_thresholds = torch.stack(synth_thresholds)
    synth_val = torch.stack([val.predictions for val in synth_val])
    full_predictions = TestSetFullPredictions(baseline=baseline_predictions, baseline_thresholds=baseline_thresholds,
                                              baseline_val=baseline_val,
                                              synth=synth_predictions, synth_thresholds=synth_thresholds,
                                              synth_val=synth_val)
    full_predictions.save_to_disk()

    log.info("-- baseline --")
    baseline_f1 = sum(metric.f1 for metric in baseline_metrics) / len(baseline_metrics)
    baseline_std = np.std([metric.f1 for metric in baseline_metrics])
    log.info(f"macro F1: {baseline_f1} +- {baseline_std}", )
    log.info(f"AUC: {sum(metric.auc for metric in baseline_metrics) / len(baseline_metrics)}")

    log.info("-- synth --")
    synth_f1 = sum(metric.f1 for metric in synth_metrics) / len(synth_metrics)
    synth_std = np.std([metric.f1 for metric in synth_metrics])
    log.info(f"macro F1: {synth_f1} +- {synth_std}", )
    log.info(f"AUC: {sum(metric.auc for metric in synth_metrics) / len(synth_metrics)}")


class TestSetFullPredictions(Artifact):
    baseline: List[TestSetPredictions]
    baseline_val: Tensor  # validation predictions stacked
    baseline_thresholds: Tensor
    synth: List[TestSetPredictions]
    synth_val: Tensor  # validation predictions stacked
    synth_thresholds: Tensor

    def generate_reports(self, test_set: TestSet):
        baseline_reports = []
        synth_reports = []

        y_true = test_set.label_tensor

        # For each set of predictions in baseline
        for thresh, test_pred in zip(self.baseline_thresholds, self.baseline):
            y_pred = test_pred.predictions > thresh
            report = classification_report(y_true, y_pred, output_dict=True, target_names=GC10_CLASSES, zero_division=0)
            baseline_reports.append(report)

        # For each set of predictions in synth
        for thresh, test_pred in zip(self.synth_thresholds, self.synth):
            y_pred = test_pred.predictions > thresh
            report = classification_report(y_true, y_pred, output_dict=True, target_names=GC10_CLASSES, zero_division=0)
            synth_reports.append(report)

        return baseline_reports, synth_reports


def pretty_print_reports(baseline_reports, synth_reports):
    # Aggregate stats across all 10 configurations
    baseline_f1_scores = [report['macro avg']['f1-score'] for report in baseline_reports]
    synth_f1_scores = [report['macro avg']['f1-score'] for report in synth_reports]

    print("Baseline:")
    print(f"Mean F1 Score: {np.mean(baseline_f1_scores):.4f}")
    print(f"Standard Deviation: {np.std(baseline_f1_scores):.4f}")

    print("Synthetic:")
    print(f"Mean F1 Score: {np.mean(synth_f1_scores):.4f}")
    print(f"Standard Deviation: {np.std(synth_f1_scores):.4f}")

    print("\nClass-wise F1 Scores:")
    for i, class_name in enumerate(GC10_CLASSES):
        baseline_class_f1_scores = [report[str(i)]['f1-score'] for report in baseline_reports]
        synth_class_f1_scores = [report[str(i)]['f1-score'] for report in synth_reports]

        print(f"Class: {class_name}")
        print(
            f"  Baseline: Mean = {np.mean(baseline_class_f1_scores):.4f}, Std = {np.std(baseline_class_f1_scores):.4f}")
        print(f"  Synthetic: Mean = {np.mean(synth_class_f1_scores):.4f}, Std = {np.std(synth_class_f1_scores):.4f}")


def plot_test_set_metrics(test_set: TestSet, val_set: ValSet, test_set_predictions: TestSetFullPredictions):
    """
        this plotting code is a mess
    @param test_set:
    @param val_set:
    @param test_set_predictions:
    @return:
    """
    prepare_latex_plot()
    fig, axes = plt.subplots(1, 2, figsize=(0.36 * 20.8, 0.36 * 5.3))

    for metric_name, ax in zip(['f1', 'auc'], axes):
        ax.grid(axis='y')  # Enable grid
        all_metrics = []
        all_val_metrics = []
        names = ['baseline', 'synth']
        base_colors = sns.color_palette("deep", 2)  # Assuming 2 main categories: Baseline, Synth
        # Desaturate for validation
        # val_colors = sns.desaturate(base_colors[0], 0.4), sns.desaturate(base_colors[1], 0.4)
        val_colors = tuple(list(base_colors[0]) + [0.7]), tuple(list(base_colors[1]) + [0.7])
        # Merge to a final palette
        color_list = [val_colors[0], base_colors[0], val_colors[1], base_colors[1]]

        for name in names:
            # Test set metrics
            macro_metrics = []
            pred_list = test_set_predictions.baseline if name == 'baseline' else test_set_predictions.synth
            threshold_vec = test_set_predictions.baseline_thresholds if name == 'baseline' \
                else test_set_predictions.synth_thresholds

            for threshold, pred in zip(threshold_vec, pred_list):  # test data
                test_metrics = f1_and_auc_on_test_set(test_labels=test_set.label_tensor, threshold_vec=threshold,
                                                      test_predictions=pred.predictions, bootstrap_m=1)
                macro_metrics.append(getattr(test_metrics, metric_name))

            all_metrics.append(macro_metrics)

            # Validation set metrics
            val_pred = test_set_predictions.baseline_val if name == 'baseline' else test_set_predictions.synth_val
            val_threshold = test_set_predictions.baseline_thresholds if name == 'baseline' \
                else test_set_predictions.synth_thresholds

            val_metrics = []
            for threshold, pred in zip(val_threshold, val_pred):  # validation data
                test_metrics = f1_and_auc_on_test_set(test_labels=val_set.label_tensor, threshold_vec=threshold,
                                                      test_predictions=pred, bootstrap_m=1)
                val_metrics.append(getattr(test_metrics, metric_name))

            all_val_metrics.append(val_metrics)

        # Plotting
        data_list = []
        for name, test_metrics in zip(names, all_metrics):
            for metric in test_metrics:
                data_list.append({'name': f'Test {name}', metric_name: metric})

        for name, test_metrics in zip(names, all_val_metrics):
            for metric in test_metrics:
                data_list.append({'name': f'Val {name}', metric_name: metric})

        df = pd.DataFrame(data_list)
        custom_order = ['Val baseline', 'Test baseline', 'Val synth', 'Test synth']

        # Convert the 'name' column to a categorical variable with custom order
        df['name'] = pd.Categorical(df['name'], categories=custom_order, ordered=True)

        # Now sort by 'name'
        df.sort_values('name', inplace=True)

        sns.swarmplot(x='name', y=metric_name, data=df, dodge=False, palette=color_list, ax=ax)

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(['Val.', 'Test', 'Val.', 'Test'], minor=False)

        def draw_errorbars(j, metrics):
            metric_mean = np.mean(metrics)
            metric_std = np.std(metrics)

            # Draw the mean line
            ax.hlines(metric_mean, xmin=j - 0.2, xmax=j + 0.2, colors=color_list[j], linewidth=2, zorder=5)

            # Draw std/errorbar with whiskers
            lower_std = metric_mean - metric_std
            upper_std = metric_mean + metric_std
            ax.errorbar(j, metric_mean, yerr=[[metric_mean - lower_std], [upper_std - metric_mean]], fmt='none',
                        color='gray', linewidth=1, capsize=5, zorder=4)
            ax.set_xlabel('')
            ax.set_ylabel(f"{'Macro-Average F1' if metric_name == 'f1' else 'AUC'}")
            # if not metric_name == 'f1':
            ax.yaxis.tick_right()

        for i, (val_metrics, test_metrics) in enumerate(zip(all_val_metrics, all_metrics)):
            draw_errorbars(2 * i, val_metrics)
            draw_errorbars(2 * i + 1, test_metrics)

        # Set the ticks on ax2 to be at the same position as on ax1,
        midpoint_baseline = 0.25
        midpoint_synthetic = 0.8
        ax2 = ax.twiny()
        ax2.xaxis.tick_bottom()
        # Setting the x-ticks for the phantom axis
        ax2.set_xticks([midpoint_baseline, midpoint_synthetic])
        ax2.set_xticklabels(['Baseline', 'Synthetic'])
        ax2.yaxis.tick_right()
        ax2.tick_params(axis='x', which='both', bottom=False, top=False)
        # Set the labels on ax2 to your high-level categories.
        # Move ax2 a bit down to not overlap with ax1.
        ax2.spines['bottom'].set_position(('outward', 20))
        ax2.xaxis.set_visible(True)
        ax2.yaxis.set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)

    plt.savefig(shared_dir.parent.parent / 'experiments' / 'figures' / 'test_set_metrics.pdf',
                bbox_inches='tight')


def read_full_predictions():
    test_full_predictions = TestSetFullPredictions.from_name('noname_44136f')
    complete_dataset = AugmentDataset.from_name('dataset_f00581')
    bundle = create_train_test_val(complete_dataset, random_seed=100, test_ratio=0.2, val_ratio=0.1, min_instances=10)

    baseline_rep, synth_rep = test_full_predictions.generate_reports(bundle.test)
    pretty_print_reports(baseline_rep, synth_rep)
    # plot_test_set_metrics(bundle.test, bundle.val, test_full_predictions)


if __name__ == '__main__':
    read_full_predictions()
    # main()
