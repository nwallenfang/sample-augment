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

figures_dir = root_dir.parent / "experiments" / "figures"


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
    num_strategies = len(config.strategies)

    reports: np.ndarray = np.empty((num_experiments, num_strategies), dtype=object)
    # baselines = None
    log.info(f"num experiments: {num_experiments}")
    #
    for i, run_name in enumerate(run_names):
        run_data = results.get(run_name, None)
        if run_data:
            report_path = root_dir / run_data[MultiSeedStrategyComparison.__full_name__]['path']
            multiseed = MultiSeedStrategyComparison.from_file(report_path)
            reports[i] = [comparison for comparison in multiseed.strategy_comparisons]
        else:
            raise NotImplementedError()
            # If MultiSeedStrategyComparison does not exist, load individual StrategyComparisonClassifiers
            # artifact_dir = root_dir / StrategyComparisonClassifiers.__name__
            # matching_files = glob.glob(f"{artifact_dir}/{run_name}_*.json", recursive=True)
            # log.info(f"loading individual StrategyComparisonClassifiers, found {len(matching_files)} comparisons")
            # reports[i]
            # for j, file in enumerate(matching_files):
            #     reports[i, j] = StrategyComparisonClassifiers.from_file(file)

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
            log.info(f"Processing report for strategy: {idx_to_name(idx)}")

            # Append macro F1 score once for the strategy
            macro_data.append({
                "Seed": seed,
                "Strategy": idx_to_name(idx),
                "Macro F1": class_report['macro avg']['f1-score']
            })
            log.debug(f"Appending macro F1 score for strategy: {idx_to_name(idx)}")
            log.info(f"Seed: {seed}, Strategy: {idx_to_name(idx)}, Macro F1: {class_report['macro avg']['f1-score']}")
            # Append individual F1 scores for each class
            for class_name in GC10_CLASSES:
                all_data.append({
                    "Seed": seed,
                    "Strategy": idx_to_name(idx),
                    "Class": class_name,
                    "F1 Score": class_report[class_name]["f1-score"]
                })
                log.debug(f"Appending F1 score for class: {class_name}")
    # df = pd.DataFrame(all_data)
    df = pd.DataFrame(macro_data)
    # replicated_data = []
    # for i in range(4):  # Four replicas
    #     new_data = df.copy()
    #     new_data['Macro F1'] = df['Macro F1'] + np.random.normal(0, 0.11, df.shape[0])  # Noise added
    #     replicated_data.append(new_data)
    # df = pd.concat(replicated_data)

    sns.set_style("whitegrid")
    # boxplot style similar to the baseline f1 comparison

    prepare_latex_plot()

    fig, ax = plt.subplots(figsize=(8, 5))
    palette = ["grey"] + list(sns.color_palette("deep", n_colors=len(report.configs['strategies'])))

    print(df.head())
    print(df['Strategy'].value_counts())
    sns.swarmplot(x='Strategy', y='Macro F1', data=df, ax=ax, hue="Strategy", palette=palette, dodge=False)

    for i, strat in enumerate(df['Strategy'].unique()):
        strat_data = df[df['Strategy'] == strat]['Macro F1']
        mean_f1 = np.mean(strat_data)
        std_f1 = np.std(strat_data)
        min_f1 = np.max([mean_f1 - std_f1, np.min(strat_data)])
        max_f1 = np.min([mean_f1 + std_f1, np.max(strat_data)])

        error_below = mean_f1 - min_f1
        error_above = max_f1 - mean_f1

        if error_below < 0:
            print(f"Clamping error_below: Original value: {error_below}, Strat: {strat}, Index: {i}")
            error_below = 0

        if error_above < 0:
            print(f"Clamping error_above: Original value: {error_above}, Strat: {strat}, Index: {i}")
            error_above = 0

        # draw means
        ax.hlines(mean_f1, xmin=i - 0.2, xmax=i + 0.2, colors=palette[i], linewidth=2)
        # draw std/errorbar with whiskers
        ax.errorbar(i, mean_f1, yerr=[[error_below], [error_above]], fmt='none', color='gray', linewidth=1,
                    capsize=5, zorder=4)

    ax.set_xticklabels(df['Strategy'].unique())
    plt.xticks(rotation=45)
    plt.ylabel('Macro Average F1 Score')

    plt.tight_layout()
    plt.savefig(figures_dir / 'multiseed_strategies_f1_swarmplot.pdf', bbox_inches="tight")


def macro_f1_for_seeds(multi_seed_report, strategies):
    """Extract the macro F1 scores for each random seed and strategy."""
    # num_strategies = len(multi_seed_report.reports[0].synth_reports)

    # Get macro F1 scores for each seed and strategy
    f1_scores_by_strategy = {}
    for i, strat in enumerate(strategies):
        f1_scores = [seed_report.synth_reports[i].report['macro avg']['f1-score']
                     for seed_report in multi_seed_report.reports]
        f1_scores_by_strategy[strat] = f1_scores

    return f1_scores_by_strategy


def mean_and_whiskers(values):
    """Compute mean and whisker values for a list of values."""
    return {
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values)
    }


def plot_macro_f1_vs_synth_p(reports):
    """Plot Macro F1 scores with whiskers against Synth P values."""
    synth_p_values = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    stats = [mean_and_whiskers(macro_f1_for_seeds(report)) for report in reports]

    means = [s["mean"] for s in stats]
    lower_errors = [s["mean"] - s["min"] for s in stats]
    upper_errors = [s["max"] - s["mean"] for s in stats]

    prepare_latex_plot()
    plt.figure(figsize=(0.9 * 6, 0.9 * 3.0))

    # Convert synth_p_values to percentages for plotting
    synth_p_percentage = [x * 100 for x in synth_p_values]

    plt.errorbar(synth_p_percentage, means, yerr=[lower_errors, upper_errors], fmt='--o', capsize=5, ecolor="grey",
                 label="Random Sampling")

    plt.xlabel(r'$\fontsize{14}{14}\selectfont p_{synth}$ (%)')
    plt.ylabel('Macro F1 Score')
    plt.legend()
    # Set y-ticks to display whole F1 scores
    min_f1 = min([s["min"] for s in stats])
    max_f1 = max([s["max"] for s in stats])
    plt.yticks(np.arange(np.floor(min_f1 * 100) / 100, np.ceil(max_f1 * 100) / 100, 0.01))

    plt.grid(True)
    # plt.show()
    plt.tight_layout()
    plt.savefig(figures_dir / "synth_p.pdf", bbox_inches="tight")


def plot_macro_f1_vs_synth_p_multistrat(reports):
    """Plot Macro F1 scores against Synth P values for multiple strategies using side-by-side subplots."""
    synth_p_values = [report.configs['synth_p'] for report in reports]
    strategies = reports[0].configs['strategies']
    prepare_latex_plot()

    fig, axs = plt.subplots(1, len(strategies), figsize=(12, 4))

    for i, strategy in enumerate(strategies):
        ax = axs[i]
        stats = [mean_and_whiskers(macro_f1_for_seeds(report, strategies)[strategy]) for report in reports]

        # means = [s["mean"] for s in stats]
        # lower_errors = [s["mean"] - s["min"] for s in stats]
        # upper_errors = [s["max"] - s["mean"] for s in stats]
        #
        # ax.errorbar(synth_p_values, means, yerr=[lower_errors, upper_errors],
        #             fmt='--o', capsize=5, ecolor="grey")
        f1_values = [macro_f1_for_seeds(report, strategies)[strategy] for report in reports]
        ax.boxplot(f1_values, notch=False, patch_artist=True)
        ax.set_xticklabels(synth_p_values)
        ax.set_xticks(range(1,
                            len(synth_p_values) + 1))  # Assuming the synth_p_values list is 1-indexed        ax.set_title(f"{strategy} Sampling")
        ax.set_xlabel(r'$p_{synth}$')
        ax.set_ylabel('Macro F1 Score')
        ax.grid(True)

    # Initialize variables
    min_f1 = float('inf')
    max_f1 = -float('inf')

    # Iterate through reports to find min and max f1 scores
    for report in reports:
        for strategy in report.configs['strategies']:
            stats = mean_and_whiskers(macro_f1_for_seeds(report, strategies)[strategy])
            min_f1 = min(min_f1, stats["min"])
            max_f1 = max(max_f1, stats["max"])

    # Then use min_f1 and max_f1 to set the y-axis limit
    for ax in axs:
        ax.set_ylim([min_f1 - 0.01, max_f1 + 0.01])
    plt.tight_layout()
    plt.savefig(figures_dir / "synth_p_multistrat_side_by_side.pdf", bbox_inches="tight")


def synth_p_lineplot():
    """Generate line plot for Macro F1 scores vs. Synth P values."""
    filenames = [
        's03-synth-p05_178d2e.json',
        's04-synth-p10_95271f.json',
        's05-synth-p15_08f3b7.json',
        's06-synth-p20_dfa894.json',
        's07-synth-p25_6250a0.json',
        's08-synth-p30_9cb4d2.json'
    ]

    # Load reports using filenames
    reports = list(map(MultiSeedReport.from_name, filenames))

    # Plot the results
    plot_macro_f1_vs_synth_p(reports)


def synth_p_detail():
    filenames = [
        's10-detail-p00_1008a0.json',
        's11-detail-p05_9441e6.json',
        's12-detail-p075_7c201a.json',
        's13-detail-p10_c06a70.json',
    ]

    # Load reports using filenames
    reports = list(map(MultiSeedReport.from_name, filenames))
    plot_macro_f1_vs_synth_p_multistrat(reports)


def get_classwise_f1_scores(reports):
    # Create a dictionary to hold arrays of F1 scores for each class
    classwise_f1 = {}

    for report in reports:
        for cls, metrics in report.items():
            if cls not in classwise_f1:
                classwise_f1[cls] = []
            classwise_f1[cls].append(metrics.get('f1-score', 0.0))

    return classwise_f1


def calc_avg_std(f1_scores):
    avg = np.mean(f1_scores)
    std = np.std(f1_scores)
    return avg, std


def compare_generators():
    cguided_old = MultiSeedReport.from_name('s12-detail-p075_7c201a.json')
    cguided_unified = MultiSeedReport.from_name('s16-unified-generator_03aa95.json')
    assert cguided_old.configs['synth_p'] == cguided_unified.configs['synth_p']
    print(cguided_old.configs['strategies'], cguided_unified.configs['strategies'])

    reports_old = [seed_report.synth_reports[1].report for seed_report in cguided_old.reports]
    reports_unified = [seed_report.synth_reports[0].report for seed_report in cguided_unified.reports]

    # Calculate class-wise F1 scores for each set of reports
    f1_old = get_classwise_f1_scores(reports_old)
    f1_unified = get_classwise_f1_scores(reports_unified)

    # Calculate and print average and std deviation for each class
    print("Class-wise F1 score comparison:")
    for cls in f1_old.keys():
        avg_old, std_old = calc_avg_std(f1_old[cls])
        avg_unified, std_unified = calc_avg_std(f1_unified.get(cls, []))

        print(f"For class {cls}:")
        print(f"  Old generator: Avg = {avg_old:.4f}, Std = {std_old:.4f}")
        print(f"  Unified generator: Avg = {avg_unified:.4f}, Std = {std_unified:.4f}")


if __name__ == '__main__':
    # strategies = MultiSeedStrategyComparison.from_name('s00-baseline_7465aa')
    # find_steps(include=['test', 'data', 'models', 'sampling'], exclude=['models.stylegan2'])
    # # create Experiment instance
    # experiment = Experiment(read_config(shared_dir / "configs/config_s00-baseline_7175d.json"))
    # experiment.run("multiseed_boxplot", initial_artifacts=[strategies])
    # multi_report = MultiSeedReport.from_name('s01-baseline_df5f64')
    # multiseed_boxplot(multi_report)
    # synth_p_lineplot()
    compare_generators()
    # synth_p_detail()
