from collections import defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sample_augment.data.dataset import AugmentDataset
from sample_augment.data.synth_data import SyntheticBundle, SynthAugmentedTrain, SynthData
from sample_augment.data.train_test_split import create_train_test_val, TrainSet, TrainTestValBundle
from sample_augment.models.evaluate_classifier import inverse_normalize
from sample_augment.models.train_classifier import train_classifier, ModelType
from sample_augment.sampling.compare_strategies import MultiSeedReport
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir


def visualize_samples(dataset, num_samples=15, times=1):
    num_rows = 3
    num_cols = num_samples // 3

    for time in range(times):
        time_offs = num_samples * time
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5))
        for i in range(num_samples):
            row = i // num_cols
            col = i % num_cols
            image, label = dataset[time_offs + row * num_cols + col]
            image = inverse_normalize(image).cpu()

            if image.shape[0] == 1:
                image = image.squeeze(0)
            else:
                image = image.permute(1, 2, 0)
            axes[row, col].imshow(image.cpu().numpy(), cmap='gray' if image.shape[0] == 1 else None)
            axes[row, col].set_title(f"Label: {label}")
            axes[row, col].axis('off')
        plt.suptitle(dataset.__full_name__)
        plt.show()
        plt.close(fig)

    # baseline_sums = torch.sum(baseline_dataset.label_tensor, dim=0)
    # synth_sums = torch.sum(synth_dataset.label_tensor, dim=0)
    # print(f'baseline: {baseline_sums}')
    # print(f'synth_sums: {synth_sums}')


def main():
    # reading synthetic data and train/test/val
    synth_bundle = SyntheticBundle.from_name('synthbundle_c83f5b')
    strategies = synth_bundle.configs['strategies']
    synth_data: SynthData = synth_bundle.synthetic_datasets[0]
    dataset = AugmentDataset.from_name('dataset_f00581')
    bundle: TrainTestValBundle = create_train_test_val(dataset, random_seed=100, test_ratio=0.2, val_ratio=0.1,
                                                       min_instances=10)
    val_data = bundle.val

    log.info(f'trying on dataset from strategy {strategies[0]}')

    # experiment constants, basically
    generator_name = 'synth-p-debug'
    generated_dir = shared_dir / "generated" / "synth-p-debug"
    synth_p = 0.0
    model_type = ModelType.VisionTransformer
    num_epochs = 120
    batch_size = 32
    learning_rate = 0.0001
    random_seed = 100
    balance_classes = True
    data_augment = True
    geometric_augment = True
    color_jitter = 0.25
    h_flip_p = 0.5
    v_flip_p = 0.5
    lr_schedule = False
    threshold_lambda = 0.4

    # defining TrainSet and SynthAugmentedTrain sets to compare them
    baseline_dataset: TrainSet = bundle.train
    synth_dataset: SynthAugmentedTrain = SynthAugmentedTrain(name=f"synth-aug-{generator_name}", root_dir=generated_dir,
                                                             img_ids=baseline_dataset.img_ids,
                                                             tensors=(
                                                                 baseline_dataset.tensors[0],
                                                                 baseline_dataset.tensors[1]),
                                                             primary_label_tensor=baseline_dataset.primary_label_tensor,
                                                             synthetic_images=synth_data.synthetic_images,
                                                             synthetic_labels=synth_data.synthetic_labels,
                                                             synth_p=synth_p,
                                                             multi_label=synth_data.multi_label)

    # need to uncomment the debug return in train_classifier() for this to work
    baseline_transformed = train_classifier(
        train_data=baseline_dataset,
        val_data=val_data,
        model_type=model_type,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        balance_classes=balance_classes,
        random_seed=random_seed,
        data_augment=data_augment,
        geometric_augment=geometric_augment,
        color_jitter=color_jitter,
        h_flip_p=h_flip_p,
        v_flip_p=v_flip_p,
        lr_schedule=lr_schedule,
        threshold_lambda=threshold_lambda
    )

    synth_transformed = train_classifier(
        train_data=synth_dataset,
        val_data=val_data,
        model_type=model_type,
        num_epochs=num_epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        balance_classes=balance_classes,
        random_seed=random_seed,
        data_augment=data_augment,
        geometric_augment=geometric_augment,
        color_jitter=color_jitter,
        h_flip_p=h_flip_p,
        v_flip_p=v_flip_p,
        lr_schedule=lr_schedule,
        threshold_lambda=threshold_lambda
    )

    # now visualize again
    # noinspection PyTypeChecker
    visualize_samples(synth_transformed, num_samples=15, times=3)
    # noinspection PyTypeChecker
    visualize_samples(baseline_transformed, num_samples=15, times=3)


class P0Stats:
    baseline_f1 = []
    strat_to_f1 = defaultdict(list)

    def add_report(self, multi_report: MultiSeedReport):
        baseline_counter = 0
        for seed_report in multi_report.reports:
            # assert seed_report.configs['synth_p'] == 0.0
            assert seed_report.configs['num_epochs'] == 120

            if seed_report.baseline_report is not None:
                baseline_counter += 1
                self.baseline_f1.append(seed_report.baseline_report.report['macro avg']['f1-score'])

            if seed_report.configs['synth_p'] == 0.0:
                for strategy, synth_report in zip(seed_report.configs['strategies'], seed_report.synth_reports):
                    self.strat_to_f1[strategy].append(synth_report.report['macro avg']['f1-score'])
            else:
                log.info(f'{multi_report.configs["name"]} - skipping synth runs')

        if baseline_counter:
            log.info(f"{multi_report.configs['name']} - added {baseline_counter} baselines")

    def print_f1(self):
        print('-- baseline --')
        print(
            f"F1: {np.mean(self.baseline_f1):.4f} +- {np.std(self.baseline_f1):.4f} ({len(self.baseline_f1)} Samples)")

        all_f1s = []
        for strat, f1s in self.strat_to_f1.items():
            print(f"-- {strat} (p_synth = 0) --")
            print(f"F1: {np.mean(f1s):.4f} +- {np.std(f1s):.4f} ({len(f1s)} Samples)")
            all_f1s.extend(f1s)

        print('-- strats merged --')
        print(f"F1: {np.mean(all_f1s):.4f} +- {np.std(all_f1s):.4f} ({len(all_f1s)} Samples)")

    def plot_f1(self):
        # noinspection PyTypeChecker
        strategy_array = (['baseline'] * len(self.baseline_f1)
                          + [strat for strat, f1s in self.strat_to_f1.items() for _ in range(len(f1s))])
        df = pd.DataFrame({
            'Strategy': strategy_array,
            'F1_Score': np.concatenate([self.baseline_f1, *self.strat_to_f1.values()])
        })

        # Create the swarmplot
        plt.figure(figsize=(10, 6))
        sns.swarmplot(x='Strategy', y='F1_Score', data=df, palette='deep', size=7)
        plt.title('F1 Score by Strategy')
        plt.xlabel('Strategy')
        plt.ylabel('F1 Score')
        plt.show()


def stats_analysis():
    multi_seed_report_baseline = MultiSeedReport.from_name('s01-baseline_df5f64.json')
    other_baselines = [
        MultiSeedReport.from_name('s03-synth-p05_178d2e.json'),
        MultiSeedReport.from_name('s04-synth-p10_95271f.json'),
        MultiSeedReport.from_name('s05-synth-p15_08f3b7.json'),
        MultiSeedReport.from_name('s06-synth-p20_dfa894.json'),
        MultiSeedReport.from_name('s07-synth-p25_6250a0.json'),
        MultiSeedReport.from_name('s08-synth-p30_9cb4d2.json')
    ]
    synth_check_multi = MultiSeedReport.from_name('s17-synth_p_0_check_9ef8df')
    detail_report = MultiSeedReport.from_name('s10-detail-p00_1008a0')
    random_strat = MultiSeedReport.from_name('s17-synth_p_0_check_random_042500.json')
    second_baseline = MultiSeedReport.from_name('s02-synth-p_f5602d.json')
    verification = MultiSeedReport.from_name('s18-synth_p_verification_3a4ea9')

    stats = P0Stats()
    stats.add_report(multi_seed_report_baseline)
    stats.add_report(synth_check_multi)
    stats.add_report(detail_report)
    stats.add_report(random_strat)
    stats.add_report(second_baseline)
    stats.add_report(verification)
    for report in other_baselines:
        stats.add_report(report)
    # random_f1s = strat_to_f1['rand']
    # classifier_guided_f1s = strat_to_f1['classifier-guided']
    stats.print_f1()
    stats.plot_f1()


if __name__ == '__main__':
    stats_analysis()
    # main()
