import sys
from typing import List

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sample_augment.core import Artifact, step
from sample_augment.data.gc10.read_labels import GC10Labels
from sample_augment.data.synth_data import SynthData, SynthAugmentedTrain, SyntheticBundle
from sample_augment.data.train_test_split import TrainSet, ValSet
from sample_augment.models.evaluate_classifier import evaluate_classifier, ClassificationReport, predict_validation_set, \
    ValidationPredictions
from sample_augment.models.generator import GC10_CLASSES
from sample_augment.models.train_classifier import ModelType
from sample_augment.models.train_classifier import train_augmented_classifier, TrainedClassifier, train_classifier
from sample_augment.sampling.classifier_guidance import classifier_guided
from sample_augment.sampling.project_images import from_projected_images
from sample_augment.sampling.random_synth import random_synthetic_augmentation
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir
from sample_augment.utils.plot import prepare_latex_plot


@step
def synth_data_to_training_set(training_set: TrainSet, synth_data: SynthData, generator_name: str, synth_p: float):
    generated_dir = shared_dir / "generated" / generator_name

    return SynthAugmentedTrain(name=f"synth-aug-{generator_name}", root_dir=generated_dir,
                               img_ids=training_set.img_ids,
                               tensors=(training_set.tensors[0], training_set.tensors[1]),
                               primary_label_tensor=training_set.primary_label_tensor,
                               synthetic_images=synth_data.synthetic_images,
                               synthetic_labels=synth_data.synthetic_labels,
                               synth_p=synth_p,
                               multi_label=synth_data.multi_label)


def hand_picked_dataset(training_set: TrainSet, generator_name: str):
    generator_name = 'handpicked-' + generator_name
    from sample_augment.sampling.synth_augment import synth_augment
    # a little redundant to do it like that but fine
    synth_aug = synth_augment(training_set=training_set, generator_name=generator_name, synth_p=0.0)

    return SynthData(synthetic_images=synth_aug.synthetic_images, synthetic_labels=synth_aug.synthetic_labels,
                     multi_label=False)


ALL_STRATEGIES = {
    "random": random_synthetic_augmentation,
    "hand-picked": hand_picked_dataset,
    "projection": from_projected_images,
    "classifier-guided": classifier_guided
}

strategy_specific_args = {
    "classifier-guided": {  # TODO pick a specific good one, this is just a random trained classifier
        "classifier": TrainedClassifier.from_name("ViT-100e_ce6b40")
    }
}


@step
def create_synthetic_bundle(strategies: List[str], training_set: TrainSet,
                            generator_name: str) -> SyntheticBundle:
    synthetic_datasets = []
    for strategy in strategies:
        log.info(f'Creating SyntheticData with strategy {strategy}')

        if strategy not in ALL_STRATEGIES:
            log.error(f'unknown strategy {strategy} provided.')
            sys.exit(-1)

        strategy_func = ALL_STRATEGIES[strategy]
        specific_args = strategy_specific_args[strategy] if strategy in strategy_specific_args else {}
        synth_set = strategy_func(training_set, generator_name, **specific_args)
        synth_set.configs['strategy'] = strategy
        synthetic_datasets.append(synth_set)

    return SyntheticBundle(synthetic_datasets=synthetic_datasets)


class StrategyComparisonClassifiers(Artifact):
    baseline: TrainedClassifier
    classifiers: List[TrainedClassifier]


@step
def synth_bundle_compare_classifiers(bundle: SyntheticBundle,
                                     strategies: List[str],
                                     train_set: TrainSet,
                                     val_set: ValSet,
                                     synth_p: float,
                                     num_epochs: int, batch_size: int, learning_rate: float,
                                     balance_classes: bool,
                                     model_type: ModelType,
                                     random_seed: int,
                                     data_augment: bool,
                                     geometric_augment: bool,
                                     color_jitter: float,
                                     h_flip_p: float,
                                     v_flip_p: float,
                                     lr_schedule: bool
                                     ) -> StrategyComparisonClassifiers:
    trained_classifiers: List[TrainedClassifier] = []
    assert len(bundle.synthetic_datasets) == len(
        strategies), f'{len(bundle.synthetic_datasets)} datasets != {len(strategies)} strategies'
    for i, synthetic_dataset in enumerate(bundle.synthetic_datasets):
        log.info(f'Training classifier with strategy {strategies[i]}.')
        # this assumes, that all datasets in the bundle used the same generator
        synth_training_set = synth_data_to_training_set(train_set, synthetic_dataset,
                                                        generator_name=bundle.configs['generator_name'],
                                                        synth_p=synth_p)
        trained_classifier = train_augmented_classifier(synth_training_set, val_set,
                                                        num_epochs, batch_size, learning_rate,
                                                        balance_classes,
                                                        model_type,
                                                        random_seed,
                                                        data_augment,
                                                        geometric_augment,
                                                        color_jitter,
                                                        h_flip_p,
                                                        v_flip_p,
                                                        lr_schedule=lr_schedule
                                                        )
        trained_classifier.configs['strategy'] = strategies[i]
        trained_classifiers.append(trained_classifier)

    log.info(f'Training baseline-configs without synthetic data.')
    baseline = train_classifier(train_data=train_set, val_data=val_set, model_type=model_type, num_epochs=num_epochs,
                                batch_size=batch_size, learning_rate=learning_rate, balance_classes=balance_classes,
                                random_seed=random_seed, data_augment=data_augment, geometric_augment=geometric_augment,
                                color_jitter=color_jitter, h_flip_p=h_flip_p, v_flip_p=v_flip_p,
                                lr_schedule=lr_schedule)
    return StrategyComparisonClassifiers(baseline=baseline, classifiers=trained_classifiers)


class SynthComparisonReport(Artifact):
    serialize_this = True
    baseline_report: ClassificationReport
    synth_reports: List[ClassificationReport]


@step
def evaluate_synth_trained_classifiers(trained_classifiers: StrategyComparisonClassifiers, val_set: ValSet,
                                       labels: GC10Labels,
                                       strategies: List[str],
                                       threshold_regularize: bool,
                                       threshold_lambda: float
                                       ) -> SynthComparisonReport:
    synth_reports = []
    for strategy, classifier in zip(strategies, trained_classifiers.classifiers):
        predictions: ValidationPredictions = predict_validation_set(classifier, val_set,
                                                                    batch_size=32)
        log.info(f"-- Strategy {strategy} --")
        report: ClassificationReport = evaluate_classifier(predictions, val_set, labels, threshold_regularize,
                                                           threshold_lambda)

        # log.info(report.report)
        synth_reports.append(report)

    log.info(" -- Baseline --")
    predictions_baseline: ValidationPredictions = predict_validation_set(trained_classifiers.baseline, val_set,
                                                                         batch_size=32)
    baseline_report: ClassificationReport = evaluate_classifier(predictions_baseline, val_set, labels,
                                                                threshold_regularize, threshold_lambda)

    return SynthComparisonReport(baseline_report=baseline_report, synth_reports=synth_reports)


@step
def create_strategy_f1_plot(synth_report: SynthComparisonReport, strategies: List[str]):
    def idx_to_name(_idx):
        return strategies[_idx - 1] if _idx > 0 else "Baseline"

    reports = [  # just poor-naming-conventions shenanigans happening here
        synth_report.baseline_report.report,
        *[synth.report for synth in synth_report.synth_reports]
    ]

    all_data = []
    for idx, report in enumerate(reports):
        for class_name in GC10_CLASSES:
            all_data.append({
                "Training Regime": idx_to_name(idx),
                "Class": class_name,
                "F1 Score": report[class_name]["f1-score"]
            })

    df = pd.DataFrame(all_data)

    plt.figure(figsize=(10, 6))
    sns.set_style("whitegrid")
    prepare_latex_plot()
    palette = ["grey"] + list(sns.color_palette("deep", n_colors=len(reports) - 1))

    # Use stripplot for the baseline-configs with '|' marker
    baseline_data = df[df["Training Regime"] == "Baseline"]
    sns.stripplot(x="F1 Score", y="Class", data=baseline_data, marker='|', jitter=False, size=15,
                  color=palette[0], linewidth=2)

    # Plot the other regimes using stripplot with 'o' marker
    for idx, regime in enumerate(df["Training Regime"].unique()[1:]):
        regime_data = df[df["Training Regime"] == regime]
        sns.stripplot(x="F1 Score", y="Class", data=regime_data, marker='o', jitter=False,
                      color=palette[idx + 1], size=8, linewidth=1.0, edgecolor="gray")

    handles = [plt.Line2D([0], [0], color=palette[i], marker=('|' if i == 0 else 'o'),
                          markersize=(15 if i == 0 else 8), linestyle='', label=idx_to_name(i),
                          linewidth=(2 if i == 0 else 1))
               for i in range(len(reports))]
    plt.legend(handles=handles, title="Strategy", loc='best')
    # plt.title("Comparison of Class-wise F1 scores for Different Training Regimes")
    plt.savefig(shared_dir / "figures" / f'strategies_f1_{synth_report.configs["name"]}.pdf', bbox_inches="tight")
