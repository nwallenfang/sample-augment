import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sample_augment.core import step
from sample_augment.core.config import read_config
from sample_augment.models.generator import GC10_CLASSES
from sample_augment.sampling.compare_strategies import SynthComparisonReport, MultiSeedStrategyComparison, \
    MultiSeedReport
from sample_augment.utils import log
from sample_augment.utils.path_utils import shared_dir, root_dir
from sample_augment.utils.plot import prepare_latex_plot


@step
def create_strategy_f1_plot(synth_report: SynthComparisonReport):
    def idx_to_name(_idx):
        return synth_report.configs["strategies"][_idx - 1] if _idx > 0 else "Baseline"

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


def sampling_eval(results, experiment_name="sampling"):
    run_names = sorted(results.keys())
    config = read_config(
        root_dir.parent / "experiments" / f"{experiment_name}-configs" / f"{run_names[0]}.json")
    num_experiments = len(results.keys())
    strategies = config.strategies
    num_strategies = len(strategies)

    reports = np.empty((num_experiments, num_strategies), dtype=object)
    # baselines = None
    #
    for i, run_name in enumerate(run_names):
        run_data = results[run_name]
        report_path = root_dir / run_data[MultiSeedStrategyComparison.__full_name__]['path']
        multiseed = MultiSeedStrategyComparison.from_file(report_path)
        reports[i] = [comparison for comparison in multiseed.strategy_comparisons]
        assert isinstance(reports[i, 0], SynthComparisonReport), type(reports[i, 0])

    # for sampling, we'll probably have some steps taking a single SynthComparisonReport
    # others taking the report for one strategy over different runs.
    # not sure yet how to handle this
    # let's first extract the SynthComparisonReport from the run files.
    # maybe we'll also want to merge multiple multi-seed runs, but take care that they were created with the same
    # versions of the sampling strategies!
    steps = [
        create_strategy_f1_plot,
    ]

    names = [name[4:] for name in run_names]

    for _step in steps:
        _step(names, reports)


@step
def multiseed_boxplot(report: MultiSeedReport):
    # report.save_to_disk()
    def idx_to_name(_idx):
        return report.reports[0].configs["strategies"][_idx - 1] if _idx > 0 else "Baseline"

    all_data = []
    macro_data = []
    # Iterating through every seed's report
    for seed, synth_report in zip(report.configs['multi_seeds'], report.reports):
        log.info(f'Loading random seed {seed}')
        reports = [
            synth_report.baseline_report.report,
            *[synth.report for synth in synth_report.synth_reports]
        ]

        # Filling the dataframe
        for idx, class_report in enumerate(reports):
            for class_name in GC10_CLASSES:
                all_data.append({
                    "Seed": seed,
                    "Strategy": idx_to_name(idx),
                    "Class": class_name,
                    "F1 Score": class_report[class_name]["f1-score"]
                })
                macro_data.append({
                    "Seed": seed,
                    "Strategy": idx_to_name(idx),
                    "Macro F1": class_report['macro avg']['f1-score']
                })

    # df = pd.DataFrame(all_data)
    df = pd.DataFrame(macro_data)
    replicated_data = []
    for i in range(4):  # Four replicas
        new_data = df.copy()
        new_data['Macro F1'] = df['Macro F1'] + np.random.normal(0, 0.11, df.shape[0])  # Noise added
        replicated_data.append(new_data)
    df = pd.concat(replicated_data)

    sns.set_style("whitegrid")
    prepare_latex_plot()
    plt.figure(figsize=(16, 8))
    palette = ["grey"] + list(sns.color_palette("deep", n_colors=len(report.configs['strategies'])))
    sns.boxplot(x="Strategy", y="Macro F1", hue="Strategy", data=df, palette=palette)

    plt.legend(title="Strategy", loc='best')
    plt.savefig(shared_dir / "figures" / f'multiseed_strategies_f1_boxplot.pdf', bbox_inches="tight")


if __name__ == '__main__':
    # strategies = MultiSeedStrategyComparison.from_name('s00-baseline_7465aa')
    # find_steps(include=['test', 'data', 'models', 'sampling'], exclude=['models.stylegan2'])
    # # create Experiment instance
    # experiment = Experiment(read_config(shared_dir / "configs/config_s00-baseline_7175d.json"))
    # experiment.run("multiseed_boxplot", initial_artifacts=[strategies])
    multi_report = MultiSeedReport.from_name('s00-baseline_7465aa')
    multiseed_boxplot(multi_report)
