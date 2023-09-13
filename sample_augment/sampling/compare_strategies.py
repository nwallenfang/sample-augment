import sys
from typing import List, Optional

from sample_augment.core import Artifact, step
from sample_augment.data.gc10.read_labels import GC10Labels
from sample_augment.data.synth_data import SynthData, SynthAugmentedTrain, SyntheticBundle
from sample_augment.data.train_test_split import TrainSet, ValSet
from sample_augment.models.evaluate_classifier import evaluate_classifier, ClassificationReport, predict_validation_set, \
    ValidationPredictions
from sample_augment.models.train_classifier import ModelType
from sample_augment.models.train_classifier import train_augmented_classifier, TrainedClassifier, train_classifier
from sample_augment.sampling.classifier_guidance import classifier_guided, GuidanceMetric
from sample_augment.sampling.project_images import from_projected_images
from sample_augment.sampling.random_synth import random_synthetic_augmentation
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir

import torch


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


def hand_picked_dataset(training_set: TrainSet, generator_name: str, _random_seed: int):
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
    "classifier-guided": classifier_guided,
    "classifier-guided-entropy": classifier_guided,
}


@step
def create_synthetic_bundle(strategies: List[str], training_set: TrainSet,
                            generator_name: str, random_seed: int) -> SyntheticBundle:
    synthetic_datasets = []
    strategy_specific_args = {
        "classifier-guided": {  # TODO pick a specific good one, this is just a random trained classifier
            "classifier": TrainedClassifier.from_name("ViT-100e_ce6b40"),
            "guidance_metric": GuidanceMetric.L2Distance
        }
    }
    for strategy in strategies:
        log.info(f'Creating SyntheticData with strategy {strategy}')

        if strategy not in ALL_STRATEGIES:
            log.error(f'unknown strategy {strategy} provided.')
            sys.exit(-1)

        strategy_func = ALL_STRATEGIES[strategy]
        specific_args = strategy_specific_args[strategy] if strategy in strategy_specific_args else {}
        synth_set = strategy_func(training_set, generator_name, random_seed, **specific_args)
        synth_set.configs['strategy'] = strategy
        synthetic_datasets.append(synth_set)

    return SyntheticBundle(synthetic_datasets=synthetic_datasets)


class StrategyComparisonClassifiers(Artifact):
    baseline: Optional[TrainedClassifier]
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
                                     lr_schedule: bool,
                                     threshold_lambda: float,
                                     train_baseline: bool = True
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
                                                        lr_schedule=lr_schedule,
                                                        threshold_lambda=threshold_lambda
                                                        )
        trained_classifier.configs['strategy'] = strategies[i]
        trained_classifier.configs['random_seed'] = random_seed
        trained_classifiers.append(trained_classifier)

    if train_baseline:
        log.info(f'Training baseline-configs without synthetic data.')
        baseline = train_classifier(train_data=train_set, val_data=val_set, model_type=model_type, num_epochs=num_epochs,
                                    batch_size=batch_size, learning_rate=learning_rate, balance_classes=balance_classes,
                                    random_seed=random_seed, data_augment=data_augment,
                                    geometric_augment=geometric_augment,
                                    color_jitter=color_jitter, h_flip_p=h_flip_p, v_flip_p=v_flip_p,
                                    lr_schedule=lr_schedule, threshold_lambda=threshold_lambda)

        # get models out of GPU so we don't run out of memory when running this for multiple in one eperiment
        baseline.model = baseline.model.cpu()
    else:
        baseline = None

    for classifier in trained_classifiers:
        classifier.model = classifier.model.cpu()
    return StrategyComparisonClassifiers(baseline=baseline, classifiers=trained_classifiers)


class MultiSeedStrategyComparison(Artifact):
    strategy_comparisons: List[StrategyComparisonClassifiers]


@step
def synth_bundle_compare_classifiers_multi_seed(
        generator_name: str,
        name: str,
        strategies: List[str],
        train_set: TrainSet,
        val_set: ValSet,
        synth_p: float,
        num_epochs: int, batch_size: int, learning_rate: float,
        balance_classes: bool,
        model_type: ModelType,
        data_augment: bool,
        geometric_augment: bool,
        color_jitter: float,
        h_flip_p: float,
        v_flip_p: float,
        lr_schedule: bool,
        threshold_lambda: float,
        multi_seeds: List[int]
) -> MultiSeedStrategyComparison:
    results = []

    for random_seed in multi_seeds:
        if artifact_name := StrategyComparisonClassifiers.exists(name=f'{name}_{random_seed}'):
            log.info(f"Seed {random_seed} already exists - skipping")
            results.append(StrategyComparisonClassifiers.from_name(artifact_name))
            continue
        log.info(f"Creating SynthBundle with random seed {random_seed}")
        bundle = create_synthetic_bundle(strategies=strategies, training_set=train_set, generator_name=generator_name,
                                         random_seed=random_seed)
        log.info(f"Running classifier training with random seed {random_seed}")
        result = synth_bundle_compare_classifiers(
            bundle,
            strategies,
            train_set,
            val_set,
            synth_p,
            num_epochs, batch_size, learning_rate,
            balance_classes,
            model_type,
            random_seed,
            data_augment,
            geometric_augment,
            color_jitter,
            h_flip_p,
            v_flip_p,
            lr_schedule,
            threshold_lambda
        )
        result.configs['name'] = f'{name}_{random_seed}'
        result.save_to_disk()
        results.append(result)

    return MultiSeedStrategyComparison(strategy_comparisons=results)


class SynthComparisonReport(Artifact):
    serialize_this = True
    baseline_report: ClassificationReport
    synth_reports: List[ClassificationReport]


@step
def evaluate_synth_trained_classifiers(trained_classifiers: StrategyComparisonClassifiers, val_set: ValSet,
                                       labels: GC10Labels,
                                       strategies: List[str],
                                       threshold_lambda: float
                                       ) -> SynthComparisonReport:
    synth_reports = []
    for idx, (strategy, classifier) in enumerate(zip(strategies, trained_classifiers.classifiers)):
        predictions: ValidationPredictions = predict_validation_set(classifier, val_set,
                                                                    batch_size=32)
        log.info(f"-- Strategy {strategy} --")
        report: ClassificationReport = evaluate_classifier(predictions, val_set, labels, threshold_lambda)

        # log.info(report.report)
        synth_reports.append(report)

    log.info(" -- Baseline --")
    predictions_baseline: ValidationPredictions = predict_validation_set(trained_classifiers.baseline, val_set,
                                                                         batch_size=32)
    baseline_report: ClassificationReport = evaluate_classifier(predictions_baseline, val_set, labels,
                                                                threshold_lambda)

    return SynthComparisonReport(baseline_report=baseline_report, synth_reports=synth_reports)


class MultiSeedReport(Artifact):
    reports: List[SynthComparisonReport]


@step
def evaluate_multiseed_synth(multiseed: MultiSeedStrategyComparison, val_set: ValSet, labels: GC10Labels,
                             strategies: List[str], threshold_lambda: float) -> MultiSeedReport:
    all_reports = []

    for seed, strategy_comparison in zip(multiseed.configs['multi_seeds'], multiseed.strategy_comparisons):
        log.info(f"-- Seed {seed} --")

        # use previously defined evaluate_synth_trained_classifiers
        synth_comparison_report = evaluate_synth_trained_classifiers(
            trained_classifiers=strategy_comparison,
            val_set=val_set,
            labels=labels,
            strategies=strategies,
            threshold_lambda=threshold_lambda
        )
        all_reports.append(synth_comparison_report)
        torch.cuda.empty_cache()

    return MultiSeedReport(reports=all_reports)
