from typing import List

import numpy as np
from matplotlib import pyplot as plt

from sample_augment.models.evaluate_classifier import ClassificationReport
from sample_augment.sampling.compare_strategies import MultiSeedReport


def plot_comparison(f1_scores_dict):
    labels = list(f1_scores_dict.keys())
    avg_scores = [np.mean(scores) for scores in f1_scores_dict.values()]
    std_scores = [np.std(scores) for scores in f1_scores_dict.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots()

    rects1 = ax.bar(x - width / 2, avg_scores, width, yerr=std_scores, label='Macro-Average F1')

    ax.set_ylabel('F1 Score')
    ax.set_title('Comparison of Candidate Configurations')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    plt.ylim(0.7, 1.0)

    fig.tight_layout()
    plt.show()


def multiseed_to_strategy_list(multiseed: MultiSeedReport, strategy_name: str) -> List[ClassificationReport]:
    idx = multiseed.configs['strategies'].index(strategy_name)
    return [seed_report.synth_reports[idx] for seed_report in multiseed.reports]


def multiseed_to_baseline_list(multiseed: MultiSeedReport) -> List[ClassificationReport]:
    return [seed_report.baseline_report for seed_report in multiseed.reports]


def determine_candiate():
    # the model config that will represent synthetic training
    #
    # A couple of ideas: Take the original c-guided and random one from the
    # synth_p = 0.33 run
    # Take the latest cguided with synth_p = 0.075
    # let's start
    reports = {
        'baseline': multiseed_to_baseline_list(MultiSeedReport.from_name('s01-baseline_df5f64')),
        'synth random': multiseed_to_strategy_list(MultiSeedReport.from_name('s01-baseline_df5f64'),
                                                   'random'),
        'synth cguided': multiseed_to_strategy_list(MultiSeedReport.from_name('s01-baseline_df5f64'),
                                                    'classifier-guided'),
        # 'new gen random':multiseed_to_strategy_list(MultiSeedReport.from_name('s16-unified-generator_03aa95'),
        #                                                     'random'),
        'new gen cguided': multiseed_to_strategy_list(MultiSeedReport.from_name('s16-unified-generator_03aa95'),
                                                      'classifier-guided'),
        # 'entropy-guided': multiseed_to_strategy_list(MultiSeedReport.from_name('s15-entropy-guidance_5d4e40'),
        #                                              'classifier-guided-entropy'),

    }

    f1_dict = {}
    for name, report in reports.items():
        f1_dict[name] = [rep.report['macro avg']['f1-score'] for rep in report]

    plot_comparison(f1_dict)


if __name__ == '__main__':
    determine_candiate()
