from typing import Optional

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


def draw_error_bars(ax, df, palette):
    for i, strat in enumerate(df['Strategy'].unique()):
        strat_data = df[df['Strategy'] == strat]['Macro F1']
        mean_f1 = np.mean(strat_data)
        print(strat, f"mean: {mean_f1}")
        std_f1 = np.std(strat_data)

        min_f1 = np.max([mean_f1 - std_f1, np.min(strat_data)])
        max_f1 = np.min([mean_f1 + std_f1, np.max(strat_data)])

        error_below = mean_f1 - min_f1
        error_above = max_f1 - mean_f1

        # draw means
        ax.hlines(mean_f1, xmin=i - 0.2, xmax=i + 0.2, colors=palette[i], linewidth=2)
        # draw std/errorbar with whiskers
        ax.errorbar(i, mean_f1, yerr=[[error_below], [error_above]], fmt='none', color='gray', linewidth=1,
                    capsize=5, zorder=4)

    ax.set_xticklabels(df['Strategy'].unique())
    # ax.xticks(rotation=45)
    ax.set_ylabel('Macro Average F1 Score')


@step
def multiseed_boxplot(report: MultiSeedReport, second_report: Optional[MultiSeedReport] = None):
    # report.save_to_disk()
    def idx_to_name(_idx, is_second=False):
        if is_second:
            return "C-guided Entropy"
        else:
            strategy_name = report.configs["strategies"][_idx - 1] if _idx > 0 else "Baseline"
            if strategy_name == "classifier-guided":
                return "C-guided $L_2$"
            return strategy_name.capitalize()

    all_data = []
    macro_data = []

    def process_report(seed, synth_report, is_second=False):
        if is_second:
            # in our case second report has no baselines trained, which is fine
            reports = [synth.report for synth in synth_report.synth_reports]
        else:
            reports = [
                synth_report.baseline_report.report,
                *[synth.report for synth in synth_report.synth_reports]
            ]

        for idx, class_report in enumerate(reports):
            macro_data.append({
                "Seed": seed,
                "Strategy": idx_to_name(idx, is_second),
                "Macro F1": class_report['macro avg']['f1-score']
            })

            for class_name in GC10_CLASSES:
                all_data.append({
                    "Seed": seed,
                    "Strategy": idx_to_name(idx, is_second),
                    "Class": class_name,
                    "F1 Score": class_report[class_name]["f1-score"]
                })

    # Process the first report
    for seed, synth_report in zip(report.configs['multi_seeds'], report.reports):
        process_report(seed, synth_report)

    # Optionally process the second report
    if second_report:
        for seed, synth_report in zip(second_report.configs['multi_seeds'], second_report.reports):
            process_report(seed, synth_report, is_second=True)

    df = pd.DataFrame(macro_data)

    sns.set_style("whitegrid")
    prepare_latex_plot()

    fig, axs = plt.subplots(1, 2, figsize=(0.55 * 11, 0.55 * 5), gridspec_kw={'width_ratios': [4, 1]})
    # Filter the dataframe to separate the 'entropy' strategy
    df_main = df[df['Strategy'] != 'C-guided Entropy']
    df_entropy = df[df['Strategy'] == 'C-guided Entropy']

    print(df_entropy)

    # Use two different palettes or you can use the same
    palette_main = ["grey"] + list(sns.color_palette("deep", n_colors=len(report.configs['strategies']) + 1))[:-1]
    palette_entropy = [list(sns.color_palette("deep", n_colors=len(report.configs['strategies']) + 1))[-1]]

    # Plot for the main strategies
    sns.swarmplot(x='Strategy', y='Macro F1', data=df_main, ax=axs[0], hue="Strategy", palette=palette_main,
                  dodge=False)
    draw_error_bars(axs[0], df_main, palette_main)
    axs[0].legend_.remove()
    axs[0].set_xlabel('')

    # Plot for the 'entropy' strategy
    sns.swarmplot(x='Strategy', y='Macro F1', data=df_entropy, ax=axs[1], hue="Strategy", palette=palette_entropy,
                  dodge=False)
    axs[1].legend_.remove()
    axs[1].set_xlabel('')
    axs[1].set_ylabel('Macro Average F1 Score')

    fig.text(0.5, 0.00, 'Strategy', ha='center', va='center')

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


def plot_macro_f1_vs_synth_p_cguided(reports):
    """Plot Macro F1 scores against Synth P values for the classifier-guided strategy."""
    synth_p_values = [report.configs['synth_p'] for report in reports]

    prepare_latex_plot()

    # Initialize a single plot
    fig, ax = plt.subplots(figsize=(6, 2.5))

    strategy = 'classifier-guided'
    strategies = reports[0].configs['strategies']
    f1_values = [macro_f1_for_seeds(report, strategies)[strategy] for report in reports]
    positions = [0, 2, 3, 4, 5]
    # Plotting the boxplot
    ax.boxplot(f1_values, positions=positions, notch=False, patch_artist=True)

    tick_labels = ['0.0', '0.05', '0.075', '0.1', '0.125']

    # Set the x-tick locations and labels
    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels)
    # ax.set_xticklabels(synth_p_values)
    # ax.set_xticks(range(1, len(synth_p_values) + 1))  # Assuming the synth_p_values list is 1-indexed
    # ax.set_title(strategy)
    ax.set_xlabel(r'$p_{synth}$', fontsize=13.5)
    ax.set_ylabel('Macro F1 Score')
    ax.grid(True)

    # Finding the min and max f1 scores to set y-axis limits
    min_f1 = min(min(f1) for f1 in f1_values)
    max_f1 = max(max(f1) for f1 in f1_values)

    ax.set_ylim([min_f1 - 0.01, max_f1 + 0.01])

    plt.tight_layout()
    plt.savefig(figures_dir / "synth_p_classifier_guided.pdf", bbox_inches="tight")


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
        's14-detail-p125_cc0c53.json'
    ]

    # Load reports using filenames
    reports = list(map(MultiSeedReport.from_name, filenames))
    plot_macro_f1_vs_synth_p_cguided(reports)


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
    multi_report = MultiSeedReport.from_name('s01-baseline_df5f64')
    # merge the entropy run into the multi_report
    entropy_run = MultiSeedReport.from_name('s15-entropy-guidance_5d4e40.json')
    # entropy_run.configs['strategies'] = ['entropy']
    # configs = multi_report.configs
    # configs["strategies"] = multi_report.configs['strategies'] + entropy_run.configs['strategies']
    # merged_report = MultiSeedReport(configs=configs, reports=[*multi_report.reports, *entropy_run.reports])
    multiseed_boxplot(multi_report, entropy_run)
    # synth_p_lineplot()
    # compare_generators()
    # synth_p_detail()

    # multi_report = MultiSeedReport.from_name('s17-synth_p_0_check_9ef8df')
    # multiseed_boxplot(multi_report)
